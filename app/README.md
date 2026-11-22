# Belot app

Bulgarian bridge-belot on Android, Windows, macOS and Linux against this
repository's bots, everything on the device. The engine is `cpp/` (`cpp/bg`
and the `cpp/belote` core it builds on); the shared app building blocks come
from [cardtable](https://github.com/frigoow/cardtable), the git submodule
`cardtable/` (`git submodule update --init`).

| Path | What |
|---|---|
| `cpp/` | `belot_api.h`: the referee, bots and hints behind a plain C interface (per-seat views as JSON, actions as ints, save/restore by replay), its JNI bridge (`jni_belot.cpp`) and tests |
| `belot/` | The table (`BelotController`: the referee's record shown in order, the bots' turns on the worker, hints, statistics, the saved game) and its screens (`BelotUi`), on cardtable's `:platform` |
| `belot-app/` | The Android shell (`com.cardtable.belot`) |
| `belot-desktop/` | The desktop shell; its tests can render the real app offscreen into PNGs |
| `models.sh` | Copies the networks from `checkpoints/` into the app's assets |
| `fetch_models.sh` | Downloads them from the release `belot-models-vN` instead |

The bots are the strongest measured configuration: the unmasked neural
bidder with doubles, card play by world sampling with the belief, fast-policy
and Q-GRU nets of the three contract kinds, racing and an exact endgame (in
Kostov's engine: 54.35% ± 1.74 pp of 2000 games against his Master). "Strong"
searches 192 worlds (about 120 ms a card on a desktop CPU without AVX2),
"Normal" 64, "Easy" plays the rule policy. To ship newer nets, export them,
point `models.sh` at their directories (`SUIT=bg_models_suit_20m app/models.sh`)
and publish them as the next `belot-models-vN`.

## Build

```bash
app/models.sh                                   # or app/fetch_models.sh
cd app
./build.sh assembleDebug                        # APK (needs the NDK)
./gradlew :belot-desktop:run                    # desktop
./gradlew -Pmanille.skipAndroidApp=true :belot:desktopTest
cmake -B build -S cpp && cmake --build build -j && ctest --test-dir build
./gradlew :belot-desktop:test -Pbelot.screenshots=/some/dir    # the app offscreen, as PNGs
```

## Release

Bump `versionName` (and `versionCode`) in `belot-app/build.gradle.kts`, push
the tag `belot-vX.Y.Z`: `.github/workflows/app-release.yml` builds the signed
APK and the three desktop packages and publishes them as that release here.
It needs the secrets `BELOT_KEYSTORE_B64`, `BELOT_KEYSTORE_PASSWORD`,
`BELOT_KEY_ALIAS` (`belot`) and `BELOT_KEY_PASSWORD`. The key is
`~/.android/belot-release.jks` on the machine that made the first release:
keep a backup, because an APK signed with any other key cannot update an
installed Belot. On macOS the version is packaged as 1.x.y while it is 0.x.y
(macOS needs 1 or more).
