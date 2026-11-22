plugins {
    id("com.android.application") version "9.1.1" apply false
    // The game and its screens live in :belot (Kotlin Multiplatform), on the
    // shared :platform of the cardbot-framework submodule; :belot-app is the Android shell.
    id("com.android.kotlin.multiplatform.library") version "9.1.1" apply false
    id("org.jetbrains.kotlin.multiplatform") version "2.3.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.3.21" apply false
    // :belot-desktop, the Windows / macOS / Linux shell around the same :belot
    id("org.jetbrains.kotlin.jvm") version "2.3.21" apply false
    id("org.jetbrains.compose") version "1.12.2" apply false
}
