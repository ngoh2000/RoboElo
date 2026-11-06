# RoboElo ratings

32 robot policies on one Elo scale, from four simulation benchmarks: RoboDojo, RoboTwin 2.0, LIBERO and SimplerEnv. The fit uses only the milestone policies ([data/milestone_policies.csv](../data/milestone_policies.csv)) that have results. How the scores are computed is described in [docs/ELO_METHOD.md](../docs/ELO_METHOD.md); the data are in [data/](../data/README.md) (public leaderboards: snapshot of 2026-10-24).

## How to read the table

- **Scale.** The reference panel (π0, π0.5, GR00T N1.7, SmolVLA, OpenVLA-OFT, X-VLA, ACT) averages 1500. On a benchmark with conversion factor a = 1, 100 points multiply the odds of success by 1.78, and 400 points by 10.
- **Overall score:** the mean of the policy's levels on the benchmarks it ran. It depends on which benchmarks those are: of two policies equally strong on LIBERO, the one that also ran a benchmark where it is weaker gets the lower score. A policy that ran one benchmark is placed against policies of other benchmarks only by the anchoring convention (method, section 5.2), which no test can confirm.
- **Full-coverage score:** the mean over all four benchmarks. A benchmark the policy did not run is filled with its predicted level, shown *(in italics)*. It is given only where the linking test shows that a benchmark the policy ran predicts the missing one (method, section 6.1); 19 of 32 policies have one. Its interval includes the error of the predicted levels.
- **Intervals:** 95 %, from 1,000 bootstrap replicates with test cases resampled within every setting. They do not include uncertainty about which policies are in the data, so cross-benchmark intervals are too narrow.
- **Levels per benchmark:** μ + γ, averaged over a benchmark's settings. "–" = not run.
- **A policy is a model recipe:** most checkpoints are fine-tuned for each benchmark (method, section 1.4).

## Policies

| # | Policy | Overall score | 95 % interval | Full-coverage score | 95 % interval | RoboDojo | RoboTwin | LIBERO | SimplerEnv |
|---|---|---|---|---|---|---|---|---|---|
| 1 | G0.5 | **1879** | 1838–1973 | 1804 | 1657–1952 | 1869 | *(1725)* | 1890 | *(1732)* |
| 2 | OpenDM0.5 | **1872** | 1822–1980 | 1798 | 1647–1949 | 1898 | *(1721)* | 1846 | *(1728)* |
| 3 | Xiaomi-R1 | **1862** | 1808–1933 | – | | 1862 | – | – | – |
| 4 | LingBot-VA | **1855** | 1813–2024 | 1741 | 1547–1935 | *(1752)* | *(1675)* | 1855 | *(1682)* |
| 5 | CogACT | **1836** | 1724–1937 | – | | – | – | – | 1836 |
| 6 | VLA-JEPA | **1786** | 1728–1892 | 1761 | 1598–1924 | *(1774)* | *(1697)* | 1835 | 1736 |
| 7 | X-VLA | **1776** | 1730–1848 | 1776 | 1730–1848 | 1721 | 1700 | 1904 | 1778 |
| 8 | π0.5 | **1767** | 1738–1838 | 1756 | 1638–1874 | 1732 | 1854 | 1717 | *(1722)* |
| 9 | hy-vla | **1763** | 1692–1843 | – | | 1763 | – | – | – |
| 10 | MolmoAct2 | **1739** | 1692–1846 | 1687 | 1537–1837 | 1676 | *(1632)* | 1801 | *(1639)* |
| 11 | InternVLA-A1.5 | **1735** | 1676–1805 | – | | 1735 | – | – | – |
| 12 | Xiaomi-R0 | **1724** | 1691–1793 | 1707 | 1588–1825 | 1655 | 1653 | 1864 | *(1655)* |
| 13 | InternVLA-M1 | **1717** | 1655–1860 | 1654 | 1462–1847 | *(1683)* | *(1606)* | 1717 | *(1613)* |
| 14 | UniVLA | **1702** | 1577–1808 | 1676 | 1499–1853 | *(1689)* | *(1612)* | 1660 | 1743 |
| 15 | ABot-M0 | **1695** | 1656–1766 | 1683 | 1564–1802 | 1540 | 1703 | 1841 | *(1648)* |
| 16 | SpatialVLA | **1648** | 1456–1818 | – | | – | – | – | 1648 |
| 17 | GR00T N1.6 | **1634** | 1466–1785 | – | | – | – | – | 1634 |
| 18 | GR00T N1.7 | **1604** | 1558–1667 | 1596 | 1501–1692 | 1495 | *(1573)* | 1670 | 1647 |
| 19 | FastWAM | **1569** | 1485–1643 | 1558 | 1426–1691 | 1540 | 1481 | 1685 | *(1528)* |
| 20 | MolmoAct | **1517** | 1478–1606 | 1540 | 1384–1697 | *(1603)* | *(1526)* | 1388 | 1645 |
| 21 | RT-1 | **1510** | 1442–1644 | – | | – | – | – | 1510 |
| 22 | starVLA | **1487** | 1436–1537 | – | | 1611 | 1364 | – | – |
| 23 | π0-FAST | **1475** | 1430–1571 | 1504 | 1316–1692 | *(1562)* | *(1485)* | 1475 | *(1492)* |
| 24 | π0 | **1425** | 1400–1468 | 1433 | 1319–1547 | 1520 | 1554 | 1202 | *(1457)* |
| 25 | OpenVLA-OFT | **1352** | 1195–1439 | 1365 | 1184–1545 | 984 | *(1374)* | 1720 | *(1381)* |
| 26 | RDT-1B | **1349** | 1265–1410 | – | | 1232 | 1467 | – | – |
| 27 | ACT | **1290** | 1098–1352 | – | | 1370 | 1209 | – | – |
| 28 | SmolVLA | **1286** | 1233–1342 | 1309 | 1167–1452 | 1438 | *(1330)* | 1133 | *(1337)* |
| 29 | RT-1-X | **1206** | 994–1344 | – | | – | – | – | 1206 |
| 30 | OpenVLA | **1163** | 1033–1286 | 1211 | 1016–1406 | *(1298)* | *(1221)* | 1248 | 1077 |
| 31 | Octo | **1100** | 600–1304 | – | | – | – | – | 1100 |
| 32 | DP | **1099** | 911–1180 | – | | – | 1099 | – | – |

## Conversion factors

How strongly a rating gap turns into success on each setting (geometric mean 1). Anchors: the policies whose mean offset on the setting is 0 (method, section 5.2).

| Setting | a | 95 % interval | Anchors |
|---|---|---|---|
| RoboDojo | 1.52 | 1.35–1.67 | π0, π0.5, X-VLA |
| LIBERO | 0.92 | 0.82–0.97 | GR00T N1.7, OpenVLA-OFT, π0, π0.5, SmolVLA, X-VLA |
| RoboTwin clean | 0.90 | 0.82–1.00 | ACT, π0, π0.5, X-VLA |
| RoboTwin randomized | 0.90 | 0.82–1.00 | ACT, π0, π0.5, X-VLA |
| SimplerEnv Google Robot | 0.79 | 0.73–0.98 | GR00T N1.7, X-VLA, MolmoAct, OpenVLA, VLA-JEPA |
| SimplerEnv WidowX | 0.79 | 0.73–0.98 | GR00T N1.7, X-VLA, OpenVLA, UniVLA, VLA-JEPA |

## Files

| File | Contents |
|---|---|
| `policy_ratings.csv` | the table above, with intervals and predicted levels |
| `settings.csv` | conversion factors with intervals, and the anchors of every setting |
| `test_difficulty.csv` | difficulty of every test case (Elo: the level that gives a 50 % chance of success) |

Reproduce: run `linking`, `fit`, `sources`, `bootstrap` and `report` of `scripts/run_unified_scale.py`, then `python scripts/write_ratings.py`.
