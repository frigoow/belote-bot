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
    private readonly AgentProcess agent;

    public RemotePlayer(IPlayer bidder, AgentProcess agent)
    {
        this.bidder = bidder;
        this.agent = agent;
    }

    public static long Fallbacks;

    public BidType GetBid(PlayerGetBidContext context) => this.bidder.GetBid(context);

    // Ours: declare everything offered (the engine keeps the first of overlapping ones).
    public IList<Announce> GetAnnounces(PlayerGetAnnouncesContext context) => context.AvailableAnnounces.ToList();

    public PlayCardAction PlayCard(PlayerPlayCardContext context)
    {
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

    public void EndOfTrick(IEnumerable<PlayCardAction> trickActions) => this.bidder.EndOfTrick(trickActions);

    public void EndOfRound(RoundResult roundResult) => this.bidder.EndOfRound(roundResult);

    public void EndOfGame(GameResult gameResult) => this.bidder.EndOfGame(gameResult);

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
    private static IPlayer Create(string name) => name switch
    {
        "master" => ClaudePlayerProfiles.CreateMaster(),
        "expert" => ClaudePlayerProfiles.CreateExpert(),
        "fast" => ClaudePlayerProfiles.CreateFast(),
        "heuristic" => ClaudePlayerProfiles.CreateHeuristic(),
        "rules" => ClaudePlayerProfiles.CreateRulesOnly(),
        "ismcts" => new ClaudePlayerIsmcts(),
        "smart" => new SmartPlayer(),
        "dummy" => new DummyPlayer(),
        _ => throw new ArgumentException($"unknown player {name}"),
    };

    public static int Main(string[] args)
    {
        string agentPath = null, agentArgs = string.Empty, opp = "master", bidder = "master";
        int pairs = 100, seed = 1, threads = Math.Max(1, Environment.ProcessorCount - 2);
        var control = false;
        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--agent": agentPath = args[++i]; break;
                case "--agent-args": agentArgs = args[++i]; break;
                case "--opp": opp = args[++i]; break;
                case "--bidder": bidder = args[++i]; break;
                case "--pairs": pairs = int.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                case "--threads": threads = int.Parse(args[++i]); break;
                case "--control": control = true; break;
                default: throw new ArgumentException($"unknown option {args[i]}");
            }
        }

        if (!control && agentPath == null)
        {
            Console.Error.WriteLine("--agent <path to bg_agent> is required (or --control)");
            return 2;
        }

        var agents = new ThreadLocal<AgentProcess>(() => control ? null : new AgentProcess(agentPath, agentArgs), true);
        var pairScore = new double[pairs];
        var pairPoints = new double[pairs];
        var sw = Stopwatch.StartNew();
        var done = 0;
        Parallel.For(0, pairs, new ParallelOptions { MaxDegreeOfParallelism = threads }, i =>
        {
            var agent = agents.Value;
            IPlayer Ours() => control ? Create(bidder) : new RemotePlayer(Create(bidder), agent);
            var first = (PlayerPosition)(1 << (i % 4));
            var dealSeed = unchecked((seed * 1000003) + i);

            // Game 1: we are South/North. Game 2: the same deals, we are East/West.
            var g1 = new BelotGame(Ours(), Create(opp), Ours(), Create(opp), new Random(dealSeed)).PlayGame(first);
            var g2 = new BelotGame(Create(opp), Ours(), Create(opp), Ours(), new Random(dealSeed)).PlayGame(first);
            var win1 = g1.SouthNorthPoints > g1.EastWestPoints ? 1.0 : 0.0;
            var win2 = g2.EastWestPoints > g2.SouthNorthPoints ? 1.0 : 0.0;
            pairScore[i] = (win1 + win2) / 2;
            pairPoints[i] = ((g1.SouthNorthPoints - g1.EastWestPoints) + (g2.EastWestPoints - g2.SouthNorthPoints)) / 2.0;
            var n = Interlocked.Increment(ref done);
            if (n % Math.Max(1, pairs / 10) == 0)
            {
                Console.Error.WriteLine($"  {n}/{pairs} pairs, {sw.Elapsed.TotalSeconds:F0}s, running {100 * pairScore.Take(n).Average():F1}%");
            }
        });

        double Mean(double[] x) => x.Average();
        double Ci(double[] x)
        {
            var m = x.Average();
            var v = x.Sum(y => (y - m) * (y - m)) / (x.Length - 1);
            return 1.96 * Math.Sqrt(v / x.Length);
        }

        var who = control ? $"{bidder} (control)" : $"ours[{agentArgs}] + {bidder} bids";
        Console.WriteLine($"{who} vs {opp}: {pairs} mirrored pairs ({2 * pairs} games), seed {seed}");
        Console.WriteLine($"  win rate {100 * Mean(pairScore):F2}% +- {100 * Ci(pairScore):F2} pp, game-point margin {Mean(pairPoints):+0.0;-0.0} +- {Ci(pairPoints):F1} per game");
        if (!control)
        {
            var calls = agents.Values.Sum(a => a.Calls);
            var ms = agents.Values.Sum(a => a.Milliseconds);
            Console.WriteLine($"  our cards {calls}, {ms / Math.Max(1, calls):F2} ms/card (round trip), fallbacks {RemotePlayer.Fallbacks}, {sw.Elapsed.TotalSeconds:F0}s");
            foreach (var a in agents.Values)
            {
                a.Dispose();
            }
        }

        return 0;
    }
}
