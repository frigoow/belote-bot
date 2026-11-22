// Arena: our C++ card player against Kostov's bots, refereed by his engine.
//
// Our seats are RemotePlayer: bids come from one of his players (for now),
// declarations and cards from our process (cpp/tools/bg_agent.cpp). Every
// pair of games uses the same deals with the teams swapped (mirrored); the
// score is the share of games won, with a 95% interval over pairs.
//
// Usage: BgBridge --agent <bg_agent> [--agent-args "..."] [--opp master] [--bidder master]
//                 [--pairs 100] [--seed 1] [--threads 8] [--control]
//   --control: our seats are the bidder's own full player (calibrates the bidder vs --opp).
//   --contract at|nt|suit: single deals instead of games; keep the mirrored pairs whose
//     contract (bid by the --bidder players on both sides) is of that kind; --pairs of them.
// Opponents: master, expert, fast, heuristic, rules, ismcts, smart, dummy.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Belot.AI.ClaudePlayer;
using Belot.AI.DummyPlayer;
using Belot.AI.SmartPlayer;
using Belot.Engine;
using Belot.Engine.Cards;
using Belot.Engine.Game;
using Belot.Engine.GameMechanics;
using Belot.Engine.Players;

public sealed class AgentProcess : IDisposable
{
    private readonly Process process;
    private readonly StreamWriter input;
    private readonly StreamReader output;

    public AgentProcess(string path, string args)
    {
        this.process = Process.Start(new ProcessStartInfo(path, args)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        });
        this.input = this.process.StandardInput;
        this.input.AutoFlush = true;
        this.output = this.process.StandardOutput;
    }

    public long Calls { get; private set; }

    public double Milliseconds { get; private set; }

    public string Ask(string line)
    {
        var sw = Stopwatch.StartNew();
        this.input.WriteLine(line);
        var reply = this.output.ReadLine();
        this.Milliseconds += sw.Elapsed.TotalMilliseconds;
        this.Calls++;
        return reply;
    }

    public void Dispose()
    {
        try
        {
            this.input.WriteLine("QUIT");
            this.process.WaitForExit(2000);
        }
        catch (IOException)
        {
        }

        this.process.Dispose();
    }
}

public sealed class RemotePlayer : IPlayer
{
    private static readonly PropertyInfo AnnounceCard =
        typeof(Announce).GetProperty("Card", BindingFlags.NonPublic | BindingFlags.Instance);

    private readonly IPlayer bidder;
    private readonly IPlayer cardPlayer;
    private readonly AgentProcess agent;
    private readonly bool remoteBid;
    private readonly bool remotePlay;

    public RemotePlayer(IPlayer bidder, IPlayer cardPlayer, AgentProcess agent, bool remoteBid, bool remotePlay)
    {
        this.bidder = bidder;
        this.cardPlayer = cardPlayer;
        this.agent = agent;
        this.remoteBid = remoteBid;
        this.remotePlay = remotePlay;
    }

    public static long Fallbacks;
    public static long BidFallbacks;

    public BidType GetBid(PlayerGetBidContext context)
    {
        if (!this.remoteBid)
        {
            return this.bidder.GetBid(context);
        }

        var sb = new StringBuilder("BID ");
        sb.Append(context.FirstToPlayInTheRound.Index()).Append(' ')
          .Append(context.MyPosition.Index()).Append(' ')
          .Append(MaskOf(context.MyCards)).Append(' ')
          .Append((int)context.AvailableBids);
        var bids = context.Bids.ToList();
        sb.Append(' ').Append(bids.Count);
        foreach (var b in bids)
        {
            sb.Append(' ').Append(b.Player.Index()).Append(' ').Append((int)b.Type);
        }

        sb.Append(' ').Append(context.SouthNorthPoints).Append(' ').Append(context.EastWestPoints)
          .Append(' ').Append(context.HangingPoints);

        var reply = this.agent.Ask(sb.ToString())?.Trim();
        if (reply != null && int.TryParse(reply, out var val))
        {
            var bt = (BidType)val;
            if (bt == BidType.Pass || context.AvailableBids.HasFlag(bt))
            {
                return bt;
            }
        }

        Interlocked.Increment(ref BidFallbacks);
        return BidType.Pass;
    }

    // Ours: declare everything offered (the engine keeps the first of overlapping ones).
    public IList<Announce> GetAnnounces(PlayerGetAnnouncesContext context) =>
        this.remotePlay ? context.AvailableAnnounces.ToList() : this.cardPlayer.GetAnnounces(context);

    public PlayCardAction PlayCard(PlayerPlayCardContext context)
    {
        if (!this.remotePlay)
        {
            return this.cardPlayer.PlayCard(context);
        }

        var sb = new StringBuilder("PLAY ");
        sb.Append(context.FirstToPlayInTheRound.Index()).Append(' ')
          .Append(context.MyPosition.Index()).Append(' ')
          .Append(MaskOf(context.MyCards)).Append(' ')
          .Append((int)context.CurrentContract.Type).Append(' ')
          .Append(context.CurrentContract.Player.Index());
        var bids = context.Bids.ToList();
        sb.Append(' ').Append(bids.Count);
        foreach (var b in bids)
        {
            sb.Append(' ').Append(b.Player.Index()).Append(' ').Append((int)b.Type);
        }

        // Declarations are public by type; their cards only for our own seat.
        var declared = context.Announces.Where(a => a.Type != AnnounceType.Belot).ToList();
        sb.Append(' ').Append(declared.Count);
        foreach (var a in declared)
        {
            var card = a.Player == context.MyPosition ? ((Card)AnnounceCard.GetValue(a)).GetHashCode() : -1;
            sb.Append(' ').Append(a.Player.Index()).Append(' ').Append((int)a.Type).Append(' ').Append(card);
        }

        var plays = context.RoundActions.ToList();
        sb.Append(' ').Append(plays.Count);
        foreach (var p in plays)
        {
            sb.Append(' ').Append(p.Player.Index()).Append(' ').Append(p.Card.GetHashCode()).Append(' ').Append(p.Belote ? 1 : 0);
        }

        sb.Append(' ').Append(context.SouthNorthPoints).Append(' ').Append(context.EastWestPoints)
          .Append(' ').Append(context.HangingPoints);

        var reply = this.agent.Ask(sb.ToString())?.Split(' ');
        Card chosen = null;
        if (reply != null && reply.Length >= 1 && int.TryParse(reply[0], out var id) && id >= 0 && id < 32)
        {
            chosen = Card.AllCards[id];
        }

        if (chosen == null || !context.AvailableCardsToPlay.Contains(chosen))
        {
            Interlocked.Increment(ref Fallbacks);
            chosen = context.AvailableCardsToPlay.FirstOrDefault();
        }

        return new PlayCardAction(chosen, true);
    }

    public void EndOfTrick(IEnumerable<PlayCardAction> trickActions)
    {
        this.bidder?.EndOfTrick(trickActions);
        if (this.cardPlayer != null && !ReferenceEquals(this.cardPlayer, this.bidder))
        {
            this.cardPlayer.EndOfTrick(trickActions);
        }
    }

    public void EndOfRound(RoundResult roundResult)
    {
        this.bidder?.EndOfRound(roundResult);
        if (this.cardPlayer != null && !ReferenceEquals(this.cardPlayer, this.bidder))
        {
            this.cardPlayer.EndOfRound(roundResult);
        }
    }

    public void EndOfGame(GameResult gameResult)
    {
        this.bidder?.EndOfGame(gameResult);
        if (this.cardPlayer != null && !ReferenceEquals(this.cardPlayer, this.bidder))
        {
            this.cardPlayer.EndOfGame(gameResult);
        }
    }

    private static uint MaskOf(IEnumerable<Card> cards)
    {
        uint m = 0;
        foreach (var c in cards)
        {
            m |= 1u << c.GetHashCode();
        }

        return m;
    }
}

public static class Program
{
    private static IPlayer CreateMasterUnlimited()
    {
        var p = ClaudePlayerProfiles.CreateMaster();
        p.SearchTimeLimitMilliseconds = 0;
        p.EndgameTimeLimitMilliseconds = 0;
        return p;
    }

    private static IPlayer Create(string name) => name switch
    {
        "master" => CreateMasterUnlimited(),
        "master-timed" => ClaudePlayerProfiles.CreateMaster(),
        "expert" => ClaudePlayerProfiles.CreateExpert(),
        "fast" => ClaudePlayerProfiles.CreateFast(),
        "heuristic" => ClaudePlayerProfiles.CreateHeuristic(),
        "rules" => ClaudePlayerProfiles.CreateRulesOnly(),
        "ismcts" => new ClaudePlayerIsmcts(),
        "smart" => new SmartPlayer(),
        "dummy" => new DummyPlayer(),
        _ => throw new ArgumentException($"unknown player {name}"),
    };

    private static IPlayer MakeOurs(string bidderName, string cardPlayerName, AgentProcess agent, bool control)
    {
        if (control)
        {
            return Create(bidderName == "ours" ? "master" : bidderName);
        }

        var remoteBid = bidderName == "ours";
        var remotePlay = cardPlayerName == "ours";
        if (!remoteBid && !remotePlay && bidderName == cardPlayerName)
        {
            return Create(bidderName);
        }

        var bidPlayer = remoteBid ? Create("master") : Create(bidderName);
        var playPlayer = remotePlay ? null : (bidderName == cardPlayerName ? bidPlayer : Create(cardPlayerName));
        return new RemotePlayer(bidPlayer, playPlayer, agent, remoteBid, remotePlay);
    }

    // Mirrored single deals of one contract kind: our margin in game points per deal.
    private static int RunDeals(string agentPath, string agentArgs, string opp, string bidder, string partnerBidder,
                                string cardPlayer, int pairs, int seed, int threads, bool control, string kind)
    {
        bool Matches(BidType c) => kind switch
        {
            "at" => c.HasFlag(BidType.AllTrumps),
            "nt" => c.HasFlag(BidType.NoTrumps),
            "suit" => c != BidType.Pass && !c.HasFlag(BidType.AllTrumps) && !c.HasFlag(BidType.NoTrumps),
            _ => throw new ArgumentException($"unknown contract kind {kind}"),
        };
        var needsRemote = !control && (bidder == "ours" || partnerBidder == "ours" || cardPlayer == "ours");
        var agents = new ThreadLocal<AgentProcess>(() => needsRemote ? new AgentProcess(agentPath, agentArgs) : null, true);
        var margins = new List<double>();
        var gate = new object();
        var next = -1;
        var tried = 0;
        var sw = Stopwatch.StartNew();
        var workers = Enumerable.Range(0, threads).Select(_ => Task.Run(() =>
        {
            var agent = agents.Value;
            IPlayer Ours0() => MakeOurs(bidder, cardPlayer, agent, control);
            IPlayer Ours1() => MakeOurs(partnerBidder, cardPlayer, agent, control);
            while (true)
            {
                lock (gate)
                {
                    if (margins.Count >= pairs)
                    {
                        return;
                    }
                }

                var i = Interlocked.Increment(ref next);
                Interlocked.Increment(ref tried);
                var first = (PlayerPosition)(1 << (i % 4));
                var dealSeed = unchecked((seed * 1000003) + i);
                var r1 = new RoundManager(Ours0(), Create(opp), Ours1(), Create(opp), new Random(dealSeed)).PlayRound(1, first, 0, 0, 0);
                if (!Matches(r1.Contract.Type))
                {
                    continue;
                }

                var r2 = new RoundManager(Create(opp), Ours0(), Create(opp), Ours1(), new Random(dealSeed)).PlayRound(1, first, 0, 0, 0);
                if (!Matches(r2.Contract.Type))
                {
                    continue;
                }

                var m = ((r1.SouthNorthPoints - r1.EastWestPoints) + (r2.EastWestPoints - r2.SouthNorthPoints)) / 2.0;
                lock (gate)
                {
                    if (margins.Count < pairs)
                    {
                        margins.Add(m);
                        if (margins.Count % Math.Max(1, pairs / 10) == 0)
                        {
                            Console.Error.WriteLine($"  {margins.Count}/{pairs} {kind} pairs ({tried} deals tried), {sw.Elapsed.TotalSeconds:F0}s, running {margins.Average():+0.00;-0.00}");
                        }
                    }
                }
            }
        })).ToArray();
        Task.WaitAll(workers);

        var mean = margins.Average();
        var ci = 1.96 * Math.Sqrt(margins.Sum(y => (y - mean) * (y - mean)) / (margins.Count - 1) / margins.Count);
        var bidLabel = bidder == partnerBidder ? $"{bidder} bids" : $"({bidder}+{partnerBidder}) bids";
        var who = control ? $"{bidder} (control)" : $"play[{cardPlayer}]({agentArgs}) + {bidLabel}";
        Console.WriteLine($"{who} vs {opp}, {kind} deals only: {margins.Count} mirrored pairs ({tried} deals dealt), seed {seed}");
        Console.WriteLine($"  game-point margin {mean:+0.00;-0.00} +- {ci:F2} per deal");
        if (needsRemote)
        {
            var calls = agents.Values.Sum(a => a.Calls);
            var ms = agents.Values.Sum(a => a.Milliseconds);
            Console.WriteLine($"  remote calls {calls}, {ms / Math.Max(1, calls):F2} ms/call (round trip), play_fallbacks {RemotePlayer.Fallbacks}, bid_fallbacks {RemotePlayer.BidFallbacks}, {sw.Elapsed.TotalSeconds:F0}s");
            foreach (var a in agents.Values)
            {
                a.Dispose();
            }
        }

        return 0;
    }

    public static int Main(string[] args)
    {
        string agentPath = null, agentArgs = string.Empty, opp = "master", bidder = "master";
        string partnerBidder = null, cardPlayer = "ours";
        int pairs = 100, seed = 1, threads = Math.Max(1, Environment.ProcessorCount - 2);
        var control = false;
        string contract = null;
        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--agent": agentPath = args[++i]; break;
                case "--agent-args": agentArgs = args[++i]; break;
                case "--opp": opp = args[++i]; break;
                case "--bidder": bidder = args[++i]; break;
                case "--partner-bidder": partnerBidder = args[++i]; break;
                case "--card-player": cardPlayer = args[++i]; break;
                case "--pairs": pairs = int.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                case "--threads": threads = int.Parse(args[++i]); break;
                case "--control": control = true; break;
                case "--contract": contract = args[++i]; break;
                default: throw new ArgumentException($"unknown option {args[i]}");
            }
        }

        partnerBidder ??= bidder;
        var needsRemote = !control && (bidder == "ours" || partnerBidder == "ours" || cardPlayer == "ours");

        if (needsRemote && agentPath == null)
        {
            Console.Error.WriteLine("--agent <path to bg_agent> is required (or --control)");
            return 2;
        }

        if (contract != null)
        {
            return RunDeals(agentPath, agentArgs, opp, bidder, partnerBidder, cardPlayer, pairs, seed, threads, control, contract);
        }

        var agents = new ThreadLocal<AgentProcess>(() => needsRemote ? new AgentProcess(agentPath, agentArgs) : null, true);
        var pairScore = new double[pairs];
        var pairPoints = new double[pairs];
        var sw = Stopwatch.StartNew();
        var done = 0;
        var doneGate = new object();
        var doneScoreSum = 0.0;
        var donePtsSum = 0.0;
        Parallel.For(0, pairs, new ParallelOptions { MaxDegreeOfParallelism = threads }, i =>
        {
            var agent = agents.Value;
            IPlayer Ours0() => MakeOurs(bidder, cardPlayer, agent, control);
            IPlayer Ours1() => MakeOurs(partnerBidder, cardPlayer, agent, control);
            var first = (PlayerPosition)(1 << (i % 4));
            var dealSeed = unchecked((seed * 1000003) + i);

            // Game 1: we are South/North. Game 2: the same deals, we are East/West.
            var g1 = new BelotGame(Ours0(), Create(opp), Ours1(), Create(opp), new Random(dealSeed)).PlayGame(first);
            var g2 = new BelotGame(Create(opp), Ours0(), Create(opp), Ours1(), new Random(dealSeed)).PlayGame(first);
            var win1 = g1.SouthNorthPoints > g1.EastWestPoints ? 1.0 : 0.0;
            var win2 = g2.EastWestPoints > g2.SouthNorthPoints ? 1.0 : 0.0;
            var ps = (win1 + win2) / 2;
            var pp = ((g1.SouthNorthPoints - g1.EastWestPoints) + (g2.EastWestPoints - g2.SouthNorthPoints)) / 2.0;
            pairScore[i] = ps;
            pairPoints[i] = pp;
            lock (doneGate)
            {
                done++;
                doneScoreSum += ps;
                donePtsSum += pp;
                if (done % Math.Max(1, pairs / 10) == 0)
                {
                    Console.Error.WriteLine($"  {done}/{pairs} pairs, {sw.Elapsed.TotalSeconds:F0}s, running {100 * doneScoreSum / done:F1}% ({donePtsSum / done:+0.0;-0.0} pts/g)");
                }
            }
        });

        double Mean(double[] x) => x.Average();
        double Ci(double[] x)
        {
            var m = x.Average();
            var v = x.Sum(y => (y - m) * (y - m)) / (x.Length - 1);
            return 1.96 * Math.Sqrt(v / x.Length);
        }

        var bidLabel = bidder == partnerBidder ? $"{bidder} bids" : $"({bidder}+{partnerBidder}) bids";
        var who = control ? $"{bidder} (control)" : $"play[{cardPlayer}]({agentArgs}) + {bidLabel}";
        Console.WriteLine($"{who} vs {opp}: {pairs} mirrored pairs ({2 * pairs} games), seed {seed}");
        Console.WriteLine($"  win rate {100 * Mean(pairScore):F2}% +- {100 * Ci(pairScore):F2} pp, game-point margin {Mean(pairPoints):+0.0;-0.0} +- {Ci(pairPoints):F1} per game");
        if (needsRemote)
        {
            var calls = agents.Values.Sum(a => a.Calls);
            var ms = agents.Values.Sum(a => a.Milliseconds);
            Console.WriteLine($"  remote calls {calls}, {ms / Math.Max(1, calls):F2} ms/call (round trip), play_fallbacks {RemotePlayer.Fallbacks}, bid_fallbacks {RemotePlayer.BidFallbacks}, {sw.Elapsed.TotalSeconds:F0}s");
            foreach (var a in agents.Values)
            {
                a.Dispose();
            }
        }

        return 0;
    }
}
