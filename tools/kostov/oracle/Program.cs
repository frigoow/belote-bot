// Oracle log for the C++ Bulgarian Belot engine (cpp/bg/): plays seeded random
// games through Kostov's BelotMatch and writes, per game,
//
//   G <game> <first seat>
//   B <round> <seat> <available BidType flags> <chosen BidType>          (asked bids only)
//   A <round> <seat> <n> <type card>... | <k> <index>...                 (offered, then declared in order)
//   P <round> <seat> <legal card mask> <card> <belote requested 0/1>     (asked cards only)
//   R <round> <first seat> <deal: 32 cards in dealing order>
//   S <round> <n> <seat type>...                                         (every bid, forced passes included)
//   C <round> <contract BidType> <contract seat>
//   N <round> <n> <seat type card scored(-1/0/1)>...                     (declared announces, belotes included)
//   T <round> <seat card belote>x4 <winner>                              (one line per trick)
//   E <round> <sn> <ew> <snTotal> <ewTotal> <noTricks> <hanging>
//   W <winner team: 0 S/N, 1 E/W> <sn> <ew> <hanging>
//
// Seats are indices S=0, E=1, N=2, W=3; cards are suit * 8 + rank (7 8 9 10 J Q K A),
// the same numbering as cpp/belote/card.h.
//
// Usage: BgOracle <games> <seed> <out-file>

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

using Belot.Engine.Cards;
using Belot.Engine.Game;
using Belot.Engine.GameMechanics;
using Belot.Engine.Players;

public static class Program
{
    private static readonly PropertyInfo AnnounceCard =
        typeof(Announce).GetProperty("Card", BindingFlags.NonPublic | BindingFlags.Instance);

    private static uint MaskOf(IEnumerable<Card> cards)
    {
        uint m = 0;
        foreach (var c in cards)
        {
            m |= 1u << c.GetHashCode();
        }

        return m;
    }

    private static int CardOf(Announce a) => ((Card)AnnounceCard.GetValue(a)).GetHashCode();

    public static int Main(string[] args)
    {
        var games = args.Length > 0 ? int.Parse(args[0]) : 100;
        var seed = args.Length > 1 ? int.Parse(args[1]) : 1;
        var path = args.Length > 2 ? args[2] : "oracle.txt";
        using var w = new StreamWriter(path);
        w.NewLine = "\n";

        for (var g = 0; g < games; g++)
        {
            var rng = new Random(unchecked((seed * 1000003) + g));
            var first = (PlayerPosition)(1 << (g % 4));
            var match = new BelotMatch(new BelotMatchOptions
            {
                FirstToPlay = first,
                Random = new Random(unchecked((seed * 7919) + g)),
                RecordHistory = true,
            });
            match.Start();
            w.WriteLine($"G {g} {first.Index()}");

            while (!match.IsFinished)
            {
                var seat = match.ToMove;
                var round = match.RoundsPlayed + 1;
                BelotAction action;
                switch (match.Decision)
                {
                    case BelotDecision.Bid:
                    {
                        var ctx = match.CreateBidContext();
                        var avail = ctx.AvailableBids;
                        var options = new List<BidType>();
                        for (var bit = 0; bit < 8; bit++)
                        {
                            var b = (BidType)(1 << bit);
                            if (avail.HasFlag(b))
                            {
                                options.Add(b);
                            }
                        }

                        var chosen = options.Count == 0 || rng.NextDouble() < 0.6
                            ? BidType.Pass
                            : options[rng.Next(options.Count)];
                        w.WriteLine($"B {round} {seat.Index()} {(int)avail} {(int)chosen}");
                        action = BelotAction.Bid(chosen);
                        break;
                    }

                    case BelotDecision.Announce:
                    {
                        var ctx = match.CreateAnnouncesContext();
                        var offered = ctx.AvailableAnnounces.ToList();
                        var order = Enumerable.Range(0, offered.Count).OrderBy(_ => rng.Next()).ToList();
                        var picked = order.Where(_ => rng.NextDouble() < 0.8).ToList();
                        var text = string.Join(" ", offered.Select(a => $"{(int)a.Type} {CardOf(a)}"));
                        w.WriteLine($"A {round} {seat.Index()} {offered.Count} {text} | {picked.Count} {string.Join(" ", picked)}");
                        action = BelotAction.Declare(picked.Select(i => offered[i]));
                        break;
                    }

                    case BelotDecision.PlayCard:
                    {
                        var ctx = match.CreatePlayCardContext();
                        var legal = ctx.AvailableCardsToPlay.ToList();
                        var card = legal[rng.Next(legal.Count)];
                        var belote = rng.NextDouble() < 0.8;
                        w.WriteLine($"P {round} {seat.Index()} {MaskOf(legal)} {card.GetHashCode()} {(belote ? 1 : 0)}");
                        action = BelotAction.PlayCard(card, belote);
                        break;
                    }

                    default:
                        throw new InvalidOperationException($"Unexpected decision {match.Decision}");
                }

                var result = match.Act(seat, action);
                if (result != BelotActResult.Ok)
                {
                    throw new InvalidOperationException($"Act failed: {result}");
                }
            }

            var record = match.GetRecord();
            foreach (var r in record.Rounds)
            {
                w.WriteLine($"R {r.RoundNumber} {r.FirstToPlay.Index()} {string.Join(" ", r.Deal.Select(c => c.GetHashCode()))}");
                w.WriteLine($"S {r.RoundNumber} {r.Bids.Count} {string.Join(" ", r.Bids.Select(b => $"{b.Player.Index()} {(int)b.Type}"))}");
                w.WriteLine($"C {r.RoundNumber} {(int)r.Contract.Type} {r.Contract.Player.Index()}");
                var anns = r.Announces ?? new List<BelotAnnounce>();
                w.WriteLine($"N {r.RoundNumber} {anns.Count} {string.Join(" ", anns.Select(a => $"{a.Player.Index()} {(int)a.Type} {a.Card?.GetHashCode() ?? -1} {(a.IsScored == null ? -1 : a.IsScored.Value ? 1 : 0)}"))}");
                foreach (var t in r.Tricks ?? new List<BelotTrick>())
                {
                    w.WriteLine($"T {r.RoundNumber} {string.Join(" ", t.Cards.Select(c => $"{c.Player.Index()} {c.Card.GetHashCode()} {(c.Belote ? 1 : 0)}"))} {t.Winner.Index()}");
                }

                var s = r.Result;
                w.WriteLine($"E {r.RoundNumber} {s.SouthNorthPoints} {s.EastWestPoints} {s.SouthNorthTotalInRoundPoints} {s.EastWestTotalInRoundPoints} {(s.NoTricksForOneOfTheTeams ? 1 : 0)} {s.HangingPoints}");
            }

            var winner = record.Winner == PlayerPosition.SouthNorthTeam ? 0 : record.Winner == PlayerPosition.EastWestTeam ? 1 : -1;
            w.WriteLine($"W {winner} {record.SouthNorthPoints} {record.EastWestPoints} {record.HangingPoints}");
        }

        return 0;
    }
}
