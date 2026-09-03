<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Design Document: TileMerge

A highly polished, addictive, single-page sliding tile puzzle game with a viral sharing hook, optimized for instant browser play and zero-configuration execution.

---

## 1. Core User Experience & The "Wow" Factor

### The Core Loop
- **Instant Play:** The user lands on a clean, modern, dark-themed page. There is zero friction—no login, no cookie banners, no ads. 
- **Juicy Feedback:** Every slide is accompanied by a fluid CSS transform transition. When two tiles merge, they trigger a subtle scale-up "pop" animation and an elegant, procedurally generated chime (using the browser's built-in **Web Audio API** so no external audio assets are needed).
- **Tactile Controls:** Fully responsive controls supporting Arrow keys, `WASD`, and smooth touch swipe gestures for mobile players.

### The Viral "Wow" Hooks
To drive organic sharing, **TileMerge** introduces two distinct viral loops:

1. **The Custom Theme Generator (URL-Shareable):**
   - Players can click a "Create Custom Theme" button to replace the standard numbers (2, 4, 8, etc.) with their own progression of 11 elements (e.g., emojis, company team members, tech stacks, or inside jokes).
   - Clicking "Generate & Share" encodes the custom theme into a compressed URL hash query parameter (e.g., `tilemerge.app/#theme=🌱,🌿,🌳,🌲,🍄,🌰,🌸,🌼,🌻,🍁,🍂` or `tilemerge.app/#theme=Intern,Junior,Mid,Senior,Lead,Staff,Principal,VP,CTO,Founder,Unicorn`).
   - When a friend clicks the link, the game immediately boots up with that custom theme, prompting them to try and reach the final "Unicorn" tile.
2. **"Wordle-Style" Result Sharing:**
   - On Win or Game Over, the player can click "Share Board".
   - This copies a beautifully formatted grid of emojis to their clipboard representing their final board layout, current score, and high tile, perfect for Twitter/X, Discord, or Slack.
   - *Example:*
     ```text
     TileMerge 1,420 pts (Reached Staff 🚀)
     🟩🟨⬛⬛
     🟩🟧🟫⬛
     🟥🟪⬛⬛
     🟦⬛⬛⬛
     Play here: [URL]
     ```

---

## 2. Concrete Feature List

### Core Gameplay & Physics
- **4x4 Grid System:** A semantic CSS grid container holding up to 16 tiles.
- **Directional Sliding:** Arrow keys and touch swipes slide all active tiles in the chosen direction (Up, Down, Left, Right).
- **Collision & Merging:**
  - Tiles with matching values merge into their sum when colliding.
  - A tile can only merge once per turn (preventing chain-collapses in a single slide).
- **Random Spawning:** After each successful move that shifts or merges at least one tile, a new tile (value of `2` with 90% probability, or `4` with 10% probability) spawns in a random empty cell with a fade-in scale animation.
- **Undo / Rewind:** A "Rewind" button allows the player to undo their last 3 moves, encouraging strategic play and reducing early frustration.

### UI & Game States
- **Scoreboard:** Live score counter displaying current points (updated on every merge by the merged tile's value).
- **Persistent High Score:** Saved in `localStorage`, persisting across browser refreshes.
- **Visual Win Overlay:** Triggered when the target tile (usually `2048`, or the 11th custom element) is created. Includes options to "Keep Playing" (sandbox mode) or "Restart".
- **Visual Game Over Overlay:** Triggered when the grid is full and no adjacent tiles share the same value (no valid moves remaining). Offers a quick-restart option.
- **Procedural Sound FX:** High-quality synth blips for slides and harmonious chords for merges, generated via the browser's Web Audio API. Can be toggled on/off with a mute button.

---

## 3. Tech Stack & Architecture

### Stack Choices
- **Frontend:** Vanilla HTML5, modern ES6+ JavaScript, and pure CSS3 with custom properties (CSS variables) for theme dynamic rendering.
- **Styling:** Tailored, lightweight custom CSS with CSS Grid and Flexbox. Tailwind CSS via CDN will be used as a utility class library for modern layout, maintaining full offline readiness by bundling fallback styles or utilizing local storage if needed.
- **State Management:** A lightweight, reactive state controller pattern in plain JS (`GameManager`, `Grid`, and `Tile` classes).

### Why This Stack is Perfect for the Sandbox
- **Zero Build Step:** There is no Webpack, Vite, or Babel compilation. The application runs immediately by opening `index.html` directly or serving it with a simple static server. This guarantees that **no build or dependency resolution issues** will occur during testing or evaluation.
- **100% Offline Capable:** The entire game runs entirely in the client's browser. It makes no external API requests, needs no external assets (audio is generated procedurally), and operates flawlessly in a completely air-gapped environment.
- **No Node/npm Friction:** While we can serve the app with any simple command, we avoid dependency bloat. It can run instantly with Python’s built-in web server or Node's zero-dependency static serves.

---

## 4. Verification and Playtesting

### Quick Launch Command
The app will be runnable using a single, built-in command from the sandbox environment:
```bash
python3 -m http.server 8000
```
or
```bash
npx http-server -p 8000
```

### Manual Verification Script / Playtest Flow
To verify that the application meets all criteria:
1. **Initialize Game:** Open `http://localhost:8000` in the browser. Verify the 4x4 grid renders cleanly with 2 initial spawned tiles, score is `0`, and the high score is retrieved from `localStorage`.
2. **Keyboard Controls:** Press Arrow keys (`Up`, `Down`, `Left`, `Right`) or `WASD`. Verify that tiles slide smoothly to the edges, matching numbers merge into their sum, and a new tile spawns on every valid move.
3. **Sound Check:** Verify a pleasing synthesized audio effect plays on slide and merge events. Click the Mute button to ensure sound can be silenced.
4. **Persisted Best Score:** Perform several moves to gain points. Refresh the browser and verify that the "Best Score" persists.
5. **Theme Generator:** Click the "Create Custom Theme" button, input custom emoji values, click "Generate & Share", load the copied URL in a new tab, and verify that the tiles now display the custom emojis instead of numbers.
6. **Game Over / Win Check:**
   - Simulate a win state or manually test to ensure reaching the 11th tile triggers the Win overlay.
   - Verify that filling the board completely with no valid moves triggers the Game Over overlay and disables sliding.
