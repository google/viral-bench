/**
 * TileMerge - Modern Sliding Tile Puzzle Game
 * Core Game Engine, Touch Event Handlers, Web Audio Synthesizer, & Custom Theme Engine.
 */

// --- 1. Sound Synthesizer via Web Audio API ---
class SoundSynth {
  constructor() {
    this.ctx = null;
    this.muted = localStorage.getItem('tilemerge_muted') === 'true';
  }

  init() {
    if (this.ctx) return;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.error("Web Audio API not supported", e);
    }
  }

  playSlide() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(180, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(320, this.ctx.currentTime + 0.08);
    
    gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.08);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    
    osc.start();
    osc.stop(this.ctx.currentTime + 0.08);
  }

  playMerge() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    // Harmonic Major chord for a super satisfying chime
    const now = this.ctx.currentTime;
    const freqs = [523.25, 659.25, 783.99]; // C5, E5, G5 (C Major)
    
    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now);
      
      // Delay play times slightly for a harp strum effect
      const delay = idx * 0.02;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(0.08, now + delay + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.25);
      
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      
      osc.start(now + delay);
      osc.stop(now + delay + 0.25);
    });
  }

  playWin() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    // Fast triumphant ascending arpeggio
    const now = this.ctx.currentTime;
    const freqs = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    
    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now + idx * 0.08);
      
      const start = now + idx * 0.08;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.12, start + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, start + 0.2);
      
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      
      osc.start(start);
      osc.stop(start + 0.2);
    });
  }

  playGameOver() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    // Descending heavy melancholic tones
    const now = this.ctx.currentTime;
    const freqs = [392.00, 349.23, 311.13]; // G4, F4, Eb4
    
    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(freq, now + idx * 0.15);
      
      const start = now + idx * 0.15;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.08, start + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, start + 0.3);
      
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      
      osc.start(start);
      osc.stop(start + 0.3);
    });
  }

  toggleMute() {
    this.muted = !this.muted;
    localStorage.setItem('tilemerge_muted', this.muted);
    return this.muted;
  }
}

const synth = new SoundSynth();


// --- 2. Themes and Customization Engine ---
const PRESETS = {
  nature: ["🌱", "🌿", "🌳", "🌲", "🍄", "🌰", "🌸", "🌼", "🌻", "🍁", "🍂"],
  dev: ["HTML", "CSS", "JS", "React", "Node", "Docker", "Python", "Rust", "AI", "Cloud", "Unicorn"],
  crypto: ["DOGE", "SHIB", "ADA", "DOT", "MATIC", "SOL", "XRP", "BNB", "ETH", "BTC", "MOON"],
  classic: ["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
};

class ThemeEngine {
  constructor() {
    this.currentTheme = [...PRESETS.classic];
    this.themeName = "Classic";
    this.loadFromUrlHash();
  }

  getTileLabel(value) {
    const idx = Math.log2(value) - 1;
    if (idx < 0) return value;
    
    // Support infinite high-scores by looping/multiplying when exceeding list length
    if (idx >= this.currentTheme.length) {
      const multiplier = Math.pow(2, idx - (this.currentTheme.length - 1));
      return `${this.currentTheme[this.currentTheme.length - 1]} x${multiplier}`;
    }
    return this.currentTheme[idx];
  }

  isClassic() {
    return this.themeName === "Classic";
  }

  applyThemeArray(arr, name = "Custom") {
    if (!Array.isArray(arr) || arr.length !== 11) return false;
    this.currentTheme = arr.map(s => s.trim().substring(0, 8) || "?");
    this.themeName = name;
    
    // Update badge in header
    const badge = document.getElementById('theme-badge');
    badge.innerText = name;
    if (name === "Classic") {
      badge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30 tracking-normal";
    } else {
      badge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/30 tracking-normal animate-pulse";
    }
    return true;
  }

  loadFromUrlHash() {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#theme=')) {
      try {
        const payload = decodeURIComponent(hash.substring(7));
        const arr = payload.split(',');
        if (arr.length === 11) {
          this.applyThemeArray(arr, "Custom");
          return true;
        }
      } catch (e) {
        console.error("Failed to decode custom theme hash link", e);
      }
    }
    return false;
  }

  generateShareUrl() {
    const encoded = encodeURIComponent(this.currentTheme.join(','));
    return `${window.location.origin}${window.location.pathname}#theme=${encoded}`;
  }
}

const themes = new ThemeEngine();


// --- 3. Tile & Grid Core Models ---
class Tile {
  static idCounter = 0;

  constructor(row, col, value) {
    this.row = row;
    this.col = col;
    this.value = value;
    this.id = Tile.idCounter++;
    this.element = null;
    this.previousPosition = { row, col };
    this.mergedInto = null; // Reference to tile this tile merged into
  }

  savePosition() {
    this.previousPosition = { row: this.row, col: this.col };
  }
}

class Grid {
  constructor(size = 4) {
    this.size = size;
    this.cells = this.emptyGrid();
  }

  emptyGrid() {
    const cells = [];
    for (let r = 0; r < this.size; r++) {
      cells.push(new Array(this.size).fill(null));
    }
    return cells;
  }

  availableCells() {
    const list = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (!this.cells[r][c]) {
          list.push({ row: r, col: c });
        }
      }
    }
    return list;
  }

  randomAvailableCell() {
    const list = this.availableCells();
    if (list.length === 0) return null;
    return list[Math.floor(Math.random() * list.length)];
  }

  insertTile(tile) {
    this.cells[tile.row][tile.col] = tile;
  }

  removeTile(tile) {
    this.cells[tile.row][tile.col] = null;
  }

  isWithinBounds(row, col) {
    return row >= 0 && row < this.size && col >= 0 && col < this.size;
  }

  getTileAt(row, col) {
    if (this.isWithinBounds(row, col)) {
      return this.cells[row][col];
    }
    return null;
  }

  hasMovesRemaining() {
    // Check if empty cells exist
    if (this.availableCells().length > 0) return true;

    // Check adjacent cells for matching values
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const tile = this.cells[r][c];
        if (tile) {
          const neighbors = [
            { r: r - 1, c: c },
            { r: r + 1, c: c },
            { r: r, c: c - 1 },
            { r: r, c: c + 1 }
          ];
          for (const n of neighbors) {
            const adj = this.getTileAt(n.r, n.c);
            if (adj && adj.value === tile.value) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  // Deep clone state for undo logic
  serialize() {
    return this.cells.map(row => 
      row.map(tile => tile ? { row: tile.row, col: tile.col, value: tile.value } : null)
    );
  }

  deserialize(state) {
    this.cells = this.emptyGrid();
    const activeTiles = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const cell = state[r][c];
        if (cell) {
          const tile = new Tile(cell.row, cell.col, cell.value);
          this.cells[r][c] = tile;
          activeTiles.push(tile);
        }
      }
    }
    return activeTiles;
  }
}


// --- 4. Game Manager (Control Loop) ---
class GameManager {
  constructor() {
    this.grid = new Grid();
    this.tiles = [];
    this.score = 0;
    this.bestScore = parseInt(localStorage.getItem('tilemerge_best') || '0', 10);
    this.won = false;
    this.gameOver = false;
    this.keepPlaying = false;
    
    // Undo Stack
    this.undoStack = [];
    this.maxUndos = 3;

    // Target Value
    this.targetValue = 2048;

    this.setupDOMElements();
    this.setupControls();
    this.newGame();
  }

  setupDOMElements() {
    this.tileContainer = document.getElementById('tile-container');
    this.scoreEl = document.getElementById('score');
    this.scoreAddEl = document.getElementById('score-addition');
    this.bestScoreEl = document.getElementById('best-score');
    
    this.overlayWin = document.getElementById('overlay-win');
    this.overlayGameOver = document.getElementById('overlay-gameover');
    this.gameoverFinalScore = document.getElementById('gameover-final-score');
    this.undoCountEl = document.getElementById('undo-count');
    this.undoBtn = document.getElementById('btn-undo');

    // Display high score on load
    this.bestScoreEl.innerText = this.bestScore;
  }

  newGame() {
    // Reset state
    this.tiles.forEach(tile => {
      if (tile.element) tile.element.remove();
    });
    this.tiles = [];
    this.grid = new Grid();
    this.score = 0;
    this.won = false;
    this.gameOver = false;
    this.keepPlaying = false;
    this.undoStack = [];
    this.updateUndoUI();

    this.scoreEl.innerText = '0';
    this.overlayWin.classList.remove('show-overlay');
    this.overlayGameOver.classList.remove('show-overlay');

    // Spawn starting tiles
    this.spawnTile();
    this.spawnTile();
    this.drawAll();
  }

  spawnTile() {
    const cell = this.grid.randomAvailableCell();
    if (cell) {
      const val = Math.random() < 0.9 ? 2 : 4;
      const tile = new Tile(cell.row, cell.col, val);
      this.grid.insertTile(tile);
      this.tiles.push(tile);
      
      // Instantiate elements
      this.createTileDOM(tile, true);
    }
  }

  createTileDOM(tile, isSpawned = false) {
    const el = document.createElement('div');
    el.className = `tile tile-${tile.value}`;
    el.style.setProperty('--row', tile.row);
    el.style.setProperty('--col', tile.col);

    const inner = document.createElement('div');
    inner.className = 'tile-inner';
    
    const label = themes.getTileLabel(tile.value);
    inner.innerText = label;

    // Apply adjustments for long words in custom themes
    if (label.length > 6) {
      inner.classList.add('custom-text-xs');
    } else if (label.length > 4) {
      inner.classList.add('custom-text-sm');
    }

    el.appendChild(inner);
    this.tileContainer.appendChild(el);
    tile.element = el;

    if (isSpawned) {
      el.classList.add('animate-spawn');
    }
  }

  drawAll() {
    // Syncs model with DOM positions
    this.tiles.forEach(tile => {
      if (tile.element) {
        tile.element.style.setProperty('--row', tile.row);
        tile.element.style.setProperty('--col', tile.col);
        // Ensure standard theme styling updates if theme resets
        tile.element.className = `tile tile-${tile.value}`;
        
        const inner = tile.element.querySelector('.tile-inner');
        if (inner) {
          const label = themes.getTileLabel(tile.value);
          inner.innerText = label;
          inner.className = 'tile-inner';
          if (label.length > 6) {
            inner.classList.add('custom-text-xs');
          } else if (label.length > 4) {
            inner.classList.add('custom-text-sm');
          }
        }
      }
    });
  }

  // Saves current checkpoint into the Undo Stack
  saveState() {
    const snapshot = {
      score: this.score,
      won: this.won,
      gameOver: this.gameOver,
      gridState: this.grid.serialize()
    };
    
    this.undoStack.push(snapshot);
    if (this.undoStack.length > this.maxUndos) {
      this.undoStack.shift();
    }
    this.updateUndoUI();
  }

  undo() {
    if (this.undoStack.length === 0) return;
    synth.init();

    const snapshot = this.undoStack.pop();
    
    // Wipe current DOM elements
    this.tiles.forEach(tile => {
      if (tile.element) tile.element.remove();
    });

    // Rebuild from history
    this.score = snapshot.score;
    this.won = snapshot.won;
    this.gameOver = snapshot.gameOver;
    
    this.scoreEl.innerText = this.score;
    this.tiles = this.grid.deserialize(snapshot.gridState);
    
    // Draw and recreate elements
    this.tiles.forEach(tile => this.createTileDOM(tile, false));
    
    // Overlays hide/show state
    if (!this.won) this.overlayWin.classList.remove('show-overlay');
    if (!this.gameOver) this.overlayGameOver.classList.remove('show-overlay');

    this.updateUndoUI();
    synth.playSlide();
  }

  updateUndoUI() {
    this.undoCountEl.innerText = this.undoStack.length;
    this.undoBtn.disabled = this.undoStack.length === 0;
  }

  // Sliders mechanics
  slide(direction) {
    if (this.won && !this.keepPlaying) return;
    if (this.gameOver) return;

    // Setup Audio
    synth.init();

    // 0 = Up, 1 = Right, 2 = Down, 3 = Left
    const vectors = [
      { r: -1, c: 0 }, // Up
      { r: 0, c: 1 },  // Right
      { r: 1, c: 0 },  // Down
      { r: 0, c: -1 }  // Left
    ];

    const vector = vectors[direction];
    
    // Scan orders: closer to sliding wall evaluates first
    const rows = [0, 1, 2, 3];
    const cols = [0, 1, 2, 3];
    
    if (direction === 2) rows.reverse(); // Down
    if (direction === 1) cols.reverse(); // Right

    let moved = false;
    let scoreAdded = 0;

    // Reset temporary merge variables
    this.tiles.forEach(tile => {
      tile.savePosition();
      tile.mergedInto = null;
    });

    // Cache current state for undo *before* calculating movement
    const preMoveSnapshot = {
      score: this.score,
      won: this.won,
      gameOver: this.gameOver,
      gridState: this.grid.serialize()
    };

    // Evaluate cell motions
    for (const r of rows) {
      for (const c of cols) {
        const tile = this.grid.getTileAt(r, c);
        if (tile) {
          let furthest = { r: tile.row, c: tile.col };
          let next = { r: furthest.r + vector.r, c: furthest.c + vector.c };
          
          // Travel in direction until boundary or block
          while (this.grid.isWithinBounds(next.r, next.c) && !this.grid.getTileAt(next.r, next.c)) {
            furthest = next;
            next = { r: furthest.r + vector.r, c: furthest.c + vector.c };
          }

          const targetTile = this.grid.getTileAt(next.r, next.c);
          
          // Collision and Merge Rules
          if (targetTile && targetTile.value === tile.value && !targetTile.mergedInto) {
            // Merge!
            this.grid.removeTile(tile);
            
            // Adjust coordinates to destination
            tile.row = targetTile.row;
            tile.col = targetTile.col;
            tile.mergedInto = targetTile;
            
            // Double the target tile
            const newValue = targetTile.value * 2;
            targetTile.value = newValue;
            targetTile.mergedInto = tile; // Link them to block further mergers this turn
            
            scoreAdded += newValue;
            moved = true;
          } else {
            // Normal slide to furthest blank spot
            if (furthest.r !== tile.row || furthest.c !== tile.col) {
              this.grid.removeTile(tile);
              tile.row = furthest.r;
              tile.col = furthest.c;
              this.grid.insertTile(tile);
              moved = true;
            }
          }
        }
      }
    }

    if (moved) {
      // Commit the preMoveSnapshot to Undo history now that we confirmed a move occurred
      this.undoStack.push(preMoveSnapshot);
      if (this.undoStack.length > this.maxUndos) {
        this.undoStack.shift();
      }
      this.updateUndoUI();

      // Clean and trigger transition merges
      const remainingTiles = [];
      let playMergeSound = false;

      this.tiles.forEach(tile => {
        if (tile.mergedInto) {
          // Slide animating to target
          tile.element.style.setProperty('--row', tile.row);
          tile.element.style.setProperty('--col', tile.col);
          
          // Sweep element out of DOM after transition completes
          const el = tile.element;
          setTimeout(() => el.remove(), 140);
          playMergeSound = true;
        } else {
          remainingTiles.push(tile);
          
          // Apply sliding animations to standard tiles
          tile.element.style.setProperty('--row', tile.row);
          tile.element.style.setProperty('--col', tile.col);
          
          // If this tile itself was the host of a merger, update its content & trigger Pop effect
          if (tile.mergedInto !== null || this.tiles.some(t => t.mergedInto === tile)) {
            setTimeout(() => {
              tile.element.className = `tile tile-${tile.value} animate-merge`;
              const inner = tile.element.querySelector('.tile-inner');
              if (inner) {
                const label = themes.getTileLabel(tile.value);
                inner.innerText = label;
                inner.className = 'tile-inner';
                if (label.length > 6) {
                  inner.classList.add('custom-text-xs');
                } else if (label.length > 4) {
                  inner.classList.add('custom-text-sm');
                }
              }
            }, 100);
          }
        }
      });

      this.tiles = remainingTiles;

      // Handle scores additions
      if (scoreAdded > 0) {
        this.score += scoreAdded;
        this.scoreEl.innerText = this.score;
        
        // Show adding visual float-up
        this.scoreAddEl.innerText = `+${scoreAdded}`;
        this.scoreAddEl.classList.remove('animate-score-add');
        void this.scoreAddEl.offsetWidth; // Force CSS reflow
        this.scoreAddEl.classList.add('animate-score-add');

        if (this.score > this.bestScore) {
          this.bestScore = this.score;
          this.bestScoreEl.innerText = this.bestScore;
          localStorage.setItem('tilemerge_best', this.bestScore);
        }
      }

      // Spawn next tile after 120ms to allow movement to complete
      setTimeout(() => {
        this.spawnTile();
        
        if (playMergeSound) {
          synth.playMerge();
        } else {
          synth.playSlide();
        }

        this.checkWinGameOver();
      }, 120);

    }
  }

  checkWinGameOver() {
    // Check Win
    if (!this.won && !this.keepPlaying) {
      const achievedTarget = this.tiles.some(tile => tile.value >= this.targetValue);
      if (achievedTarget) {
        this.won = true;
        setTimeout(() => {
          this.overlayWin.classList.add('show-overlay');
          synth.playWin();
        }, 300);
      }
    }

    // Check Game Over
    if (!this.grid.hasMovesRemaining()) {
      this.gameOver = true;
      this.gameoverFinalScore.innerText = this.score;
      setTimeout(() => {
        this.overlayGameOver.classList.add('show-overlay');
        synth.playGameOver();
      }, 300);
    }
  }

  setupControls() {
    // 1. Keyboard Listeners
    window.addEventListener('keydown', (e) => {
      // Prevent scrolling
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
        e.preventDefault();
      }

      switch (e.code) {
        case 'ArrowUp':
        case 'KeyW':
          this.slide(0);
          break;
        case 'ArrowRight':
        case 'KeyD':
          this.slide(1);
          break;
        case 'ArrowDown':
        case 'KeyS':
          this.slide(2);
          break;
        case 'ArrowLeft':
        case 'KeyA':
          this.slide(3);
          break;
      }
    });

    // 2. Mobile Touch Swipe Handlers
    let touchStartX = 0;
    let touchStartY = 0;
    const swipeThreshold = 40; // minimum slide distance in pixels

    const board = document.getElementById('game-container');
    
    board.addEventListener('touchstart', (e) => {
      if (e.touches.length > 1) return; // ignore multitouch
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
    }, { passive: true });

    board.addEventListener('touchend', (e) => {
      if (e.changedTouches.length === 0) return;
      
      const dx = e.changedTouches[0].clientX - touchStartX;
      const dy = e.changedTouches[0].clientY - touchStartY;
      
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);

      if (Math.max(absDx, absDy) > swipeThreshold) {
        if (absDx > absDy) {
          // Horizontal Swipe
          if (dx > 0) {
            this.slide(1); // Right
          } else {
            this.slide(3); // Left
          }
        } else {
          // Vertical Swipe
          if (dy > 0) {
            this.slide(2); // Down
          } else {
            this.slide(0); // Up
          }
        }
      }
    }, { passive: true });

    // 3. Game Control Buttons
    document.getElementById('btn-restart').onclick = () => this.newGame();
    document.getElementById('btn-undo').onclick = () => this.undo();
    
    document.getElementById('btn-win-continue').onclick = () => {
      this.keepPlaying = true;
      this.overlayWin.classList.remove('show-overlay');
    };
    document.getElementById('btn-win-restart').onclick = () => this.newGame();
    document.getElementById('btn-gameover-restart').onclick = () => this.newGame();

    // 4. Sound Control Toggle
    const btnSound = document.getElementById('btn-sound');
    const iconOn = document.getElementById('sound-icon-on');
    const iconOff = document.getElementById('sound-icon-off');

    const renderMuteState = (isMuted) => {
      if (isMuted) {
        iconOn.classList.add('hidden');
        iconOff.classList.remove('hidden');
      } else {
        iconOn.classList.remove('hidden');
        iconOff.classList.add('hidden');
      }
    };

    renderMuteState(synth.muted);

    btnSound.onclick = () => {
      const isMuted = synth.toggleMute();
      renderMuteState(isMuted);
      showToast(isMuted ? "Audio muted" : "Audio unmuted", "🔊");
    };

    // 5. Custom Theme Creator UI
    const themeModal = document.getElementById('theme-modal');
    const openThemeBtn = document.getElementById('btn-open-theme-creator');
    const closeThemeBtn = document.getElementById('btn-close-theme-modal');
    const applyThemeBtn = document.getElementById('btn-apply-theme');
    const resetThemeCreatorBtn = document.getElementById('btn-reset-theme-creator');

    openThemeBtn.onclick = () => {
      themeModal.classList.add('show-overlay');
      // Populate modal input values with current theme items
      const inputs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
      inputs.forEach((val) => {
        const inputEl = document.getElementById(`tile-input-${val}`);
        if (inputEl) {
          inputEl.value = themes.getTileLabel(val);
        }
      });
    };

    closeThemeBtn.onclick = () => {
      themeModal.classList.remove('show-overlay');
    };

    applyThemeBtn.onclick = () => {
      const inputs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
      const customValues = inputs.map(val => {
        const v = document.getElementById(`tile-input-${val}`).value.trim();
        return v || "?";
      });

      themes.applyThemeArray(customValues, "Custom");
      this.drawAll();
      
      // Copy share URL to clipboard automatically on custom creation
      const shareUrl = themes.generateShareUrl();
      navigator.clipboard.writeText(shareUrl).then(() => {
        showToast("Theme loaded & share link copied!", "🎨");
      }).catch(() => {
        showToast("Custom theme applied!", "🎨");
      });

      themeModal.classList.remove('show-overlay');
    };

    resetThemeCreatorBtn.onclick = () => {
      themes.applyThemeArray([...PRESETS.classic], "Classic");
      this.drawAll();
      window.location.hash = ""; // Clear share hash
      showToast("Theme reset to Classic", "🔄");
      themeModal.classList.remove('show-overlay');
    };

    // Preset shortcut click handlers inside Custom Creator modal
    document.getElementById('btn-preset-nature').onclick = () => this.applyPreset('nature');
    document.getElementById('btn-preset-dev').onclick = () => this.applyPreset('dev');
    document.getElementById('btn-preset-crypto').onclick = () => this.applyPreset('crypto');

    // 6. Board Result Sharing (Wordle style)
    document.getElementById('btn-share-win').onclick = () => this.shareBoardResult();
    document.getElementById('btn-share-gameover').onclick = () => this.shareBoardResult();
  }

  applyPreset(presetKey) {
    const arr = PRESETS[presetKey];
    if (arr) {
      const inputs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
      inputs.forEach((val, idx) => {
        const inputEl = document.getElementById(`tile-input-${val}`);
        if (inputEl) {
          inputEl.value = arr[idx];
        }
      });
    }
  }

  // Generates Wordle-style copy-to-clipboard stats
  shareBoardResult() {
    let text = `TileMerge ${this.score.toLocaleString()} pts (Goal reached: ${this.won ? 'YES 🏆' : 'NO 💀'})\n`;
    
    // Determine highest tile
    let maxVal = 2;
    this.tiles.forEach(tile => { if (tile.value > maxVal) maxVal = tile.value; });
    text += `Highest Tile: ${themes.getTileLabel(maxVal)}\n\n`;

    // Map 4x4 board to dynamic visual emojis
    // We categorize custom words or numbers to distinct high-contrast colors
    const colorEmojiMap = {
      2: "⬜", 4: "🟪", 8: "🟫", 16: "🟧", 32: "🟥", 
      64: "pink", 128: "🟨", 256: "pink", 512: "🟦", 
      1024: "cyan", 2048: "🚀"
    };

    for (let r = 0; r < 4; r++) {
      let rowStr = "";
      for (let c = 0; c < 4; c++) {
        const tile = this.grid.getTileAt(r, c);
        if (!tile) {
          rowStr += "⬛"; // Empty
        } else {
          // Use high-contrast grid maps based on values
          const val = tile.value;
          if (val === 2) rowStr += "⬜";
          else if (val === 4) rowStr += "🟪";
          else if (val === 8) rowStr += "🟫";
          else if (val === 16) rowStr += "🟧";
          else if (val === 32) rowStr += "🟥";
          else if (val === 64) rowStr += "🟫";
          else if (val === 128) rowStr += "🟨";
          else if (val === 256) rowStr += "🟩";
          else if (val === 512) rowStr += "🟦";
          else if (val === 1024) rowStr += "🌐";
          else rowStr += "🚀"; // 2048+
        }
      }
      text += rowStr + "\n";
    }

    const shareUrl = themes.isClassic() ? window.location.origin + window.location.pathname : themes.generateShareUrl();
    text += `\nPlay custom puzzle link:\n${shareUrl}`;

    navigator.clipboard.writeText(text).then(() => {
      showToast("Stats & board layout copied!", "📋");
    }).catch(() => {
      showToast("Failed to copy stats", "❌");
    });
  }
}


// --- 5. Global Helpers & Init ---

// Global Notification Toast
function showToast(message, icon = "✨") {
  const toast = document.getElementById('toast');
  const iconEl = document.getElementById('toast-icon');
  const msgEl = document.getElementById('toast-message');

  iconEl.innerText = icon;
  msgEl.innerText = message;

  toast.classList.add('show-overlay');
  toast.classList.remove('scale-90', 'translate-y-2', 'opacity-0');

  // Clear previous timer if exists
  if (window.toastTimer) {
    clearTimeout(window.toastTimer);
  }

  window.toastTimer = setTimeout(() => {
    toast.classList.remove('show-overlay');
    toast.classList.add('scale-90', 'translate-y-2', 'opacity-0');
  }, 2500);
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
  const game = new GameManager();
  
  // Audio Opt-In initializer overlay: clicking anywhere starts/unmutes audio elements
  const optIn = document.getElementById('audio-opt-in');
  const triggerAudioInit = () => {
    synth.init();
    optIn.removeEventListener('click', triggerAudioInit);
    document.removeEventListener('keydown', triggerAudioInit);
  };
  optIn.addEventListener('click', triggerAudioInit);
  document.addEventListener('keydown', triggerAudioInit);
});
