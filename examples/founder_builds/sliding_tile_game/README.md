# TileMerge 🧩

An addictive, beautiful, and highly polished single-page 2048-style sliding tile puzzle game. Zero install, zero dependencies, and fully optimized for quick sessions in desktop or mobile browsers.

---

## 🌟 Viral "Wow" Features

TileMerge is engineered to go viral by lowering entry barriers to zero and offering engaging, native sharing hooks:

1. **🎨 Dynamic URL Custom Themes:**
   - Create your own customized set of 11 sliding emojis or text stages (e.g., Tech Stacks, Inside Jokes, Food progression, or Team members) directly in-game.
   - Click "Generate & Load Theme" to generate a compressed URL hash link that encodes your theme. Share it with friends so they can immediately play your exact game configuration!
2. **📈 Wordle-Style Clipboard Board Sharing:**
   - Copy beautifully formatted, high-contrast grid layouts showing your final board, score, and highest tiles directly to your clipboard for quick pasting on Twitter/X, Discord, Slack, or SMS.
3. **🎵 Procedural Synthesizer Sound FX:**
   - Dynamic, pleasant synth chimes and effects procedurally synthesized directly via the browser's native **Web Audio API**. Works fully offline with zero asset-loading wait time!

---

## 🎮 How to Play

1. Use your **Arrow Keys** or **WASD** (or swipe on touchscreens).
2. All tiles slide together in the direction of your choice.
3. Matching tiles merge into their sum.
4. Try to reach the target tile of **2048** (or the 11th custom milestone of your custom theme)!
5. Utilize the **Undo** button to rewind your last 3 moves when making complex strategic decisions.

---

## 🛠️ Tech Stack & Architecture

- **Frontend:** HTML5, CSS3 Grid with calculated exact animations, and modern ES6+ vanilla JavaScript.
- **Styling Framework:** Tailwind CSS CDN (with fallback responsiveness).
- **Audio Synthesizer:** Pure Web Audio API oscillators and gains.
- **Offline Reliability:** Zero NPM build step, zero Node dependencies, and 100% client-side compilation-free layout ensuring absolute robustness in air-gapped runtimes.

---

## 🚀 Launch Instructions

To launch the app immediately in your development or production environment, execute a simple zero-configuration HTTP server in this directory:

### Python 3
```bash
python3 -m http.server 8000
```

### Node.js (npx)
```bash
npx http-server -p 8000
```

Once running, navigate to **`http://localhost:8000`** in any web browser to play!

---

## 🧪 Testing Checklist

- [x] **Tile Spawning:** Confirmed 2 random tiles spawn at start; a new tile (2 or 4) spawns after every valid sliding move.
- [x] **Sliding & Merging Physics:** Confirmed full 4-directional sliding behavior, correct same-value combination logic, and single-merge-per-turn rules.
- [x] **High Score Persistence:** Confirmed current score displays accurately and best score persists in local storage upon hard refresh.
- [x] **Theme Encoding:** Confirmed custom emoji arrays compile correctly into URL hash parameters and load dynamically on link access.
- [x] **Audio Playback:** Confirmed oscillator slide sweeps, chord merge rings, and triumph arpeggios play accurately when active, and can be muted.
