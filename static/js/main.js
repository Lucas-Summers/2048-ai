document.addEventListener('DOMContentLoaded', () => {
    // Game constants
    const DIRECTIONS = {
        UP: 0,
        RIGHT: 1,
        DOWN: 2,
        LEFT: 3
    };

    const SPEEDS = {
        'full': 0,
        'fast': 100,
        'medium': 300,
        'slow': 500
    };

    // Animation scaling factors for different speeds
    const ANIMATION_SCALES = {
        'full': 0.1,  // Almost instant animations
        'fast': 0.5,  // Half the time of medium
        'medium': 1,  // Base values
        'slow': 1.5   // 50% slower than medium
    };

    // Game state
    let gameId = 'default';
    let gameRunning = false;
    let board = [];
    let score = 0;
    let bestScore = localStorage.getItem('bestScore') || 0;
    let isGameOver = false;
    let aiAgent = 'random';
    let aiSpeed = 'fast';
    let tileHistory = {};  // To track highest tiles achieved
    let aiTimeoutId = null; // Track timeout ID for AI moves
    let isResetting = false; // Flag to indicate a game reset is in progress
    let currentAbortController = null; // For cancelling in-flight fetch requests

    // DOM elements
    const gridContainer = document.getElementById('grid-container');
    const scoreDisplay = document.getElementById('score');
    const bestDisplay = document.getElementById('best');
    const resumeButton = document.getElementById('resume-button');
    const newGameButton = document.getElementById('new-game-button');
    const aiAgentsRadios = document.getElementsByName('ai-mode');
    const speedRadios = document.getElementsByName('speed');
    const agentStats = document.getElementById('agent-stats');
    const highestTilesDisplay = document.getElementById('highest-tiles');
    const gameOverModal = document.getElementById('game-over-modal');
    const gameOverStats = document.getElementById('game-over-stats');
    const modalNewGameButton = document.getElementById('modal-new-game');

    // Initialize the game
    initGame();

    // Event listeners
    resumeButton.addEventListener('click', toggleAI);
    newGameButton.addEventListener('click', startNewGame);
    modalNewGameButton.addEventListener('click', () => {
        gameOverModal.style.display = 'none';
        startNewGame();
    });

    // Add event listeners to AI agent radios
    for (const radio of aiAgentsRadios) {
        radio.addEventListener('change', updateAIAgent);
    }

    // Add event listeners to speed radios
    for (const radio of speedRadios) {
        radio.addEventListener('change', updateAISpeed);
    }

    // Keyboard controls
    document.addEventListener('keydown', handleKeyPress);

    /**
     * Initialize the game
     */
    function initGame() {
        // Create grid cells
        createGameGrid();
        
        // Ensure the correct speed is selected in the UI
        updateSpeedSelection();
        
        // Update animation speeds based on the selected speed
        updateAnimationSpeeds();
        
        // Start new game
        startNewGame();
    }

    /**
     * Update animation speeds based on the selected AI speed
     * Sets CSS variables for animation timing
     * @returns {number} - Total animation duration in milliseconds
     */
    function updateAnimationSpeeds() {
        const scale = ANIMATION_SCALES[aiSpeed];
        
        // Calculate scaled durations with minimums
        const movementDuration = Math.max(20, Math.round(200 * scale));
        const mergeDelay = Math.max(10, Math.round(200 * scale));
        const newTileDelay = Math.max(10, Math.round(200 * scale));
        
        // Update CSS variables
        document.documentElement.style.setProperty('--movement-duration', `${movementDuration}ms`);
        document.documentElement.style.setProperty('--merge-delay', `${mergeDelay}ms`);
        document.documentElement.style.setProperty('--new-tile-delay', `${newTileDelay}ms`);
        
        // Calculate total animation time for scheduling purposes
        const totalAnimationTime = movementDuration + mergeDelay + 50; // 50ms buffer
        
        console.log(`Animation speeds updated for ${aiSpeed}: movement=${movementDuration}ms, merge=${mergeDelay}ms`);
        
        return totalAnimationTime;
    }

    /**
     * Create the game grid
     */
    function createGameGrid() {
        gridContainer.innerHTML = '';
        
        // Create background grid cells
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.style.left = `${col * 116.25 + 10}px`;
                cell.style.top = `${row * 116.25 + 10}px`;
                gridContainer.appendChild(cell);
            }
        }
    }

    /**
     * Start a new game
     */
    function startNewGame() {
        console.log("Starting new game...");
        
        // Set resetting flag to prevent further AI moves
        isResetting = true;
        
        // Stop the AI immediately
        if (gameRunning) {
            stopAI();
        }
        
        // Cancel any in-flight requests
        if (currentAbortController !== null) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        
        // Clear all existing timeouts that might trigger animations
        const highestTimeoutId = setTimeout(() => {}, 0);
        for (let i = 0; i < highestTimeoutId; i++) {
            clearTimeout(i);
        }
        
        // Force an immediate full state reset
        // Clear the game grid of all tiles immediately
        document.getElementById('grid-container').innerHTML = '';
        
        // Create background grid cells
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.style.left = `${col * 116.25 + 10}px`;
                cell.style.top = `${row * 116.25 + 10}px`;
                gridContainer.appendChild(cell);
            }
        }
        
        // Reset game state immediately
        board = [];
        score = 0;
        isGameOver = false;
        tileHistory = {};
        
        // Update the score display immediately
        scoreDisplay.textContent = score;
        highestTilesDisplay.innerHTML = '';
        
        // Now fetch the new game state from the server
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_id: gameId })
        })
        .then(response => response.json())
        .then(data => {
            // Update game state with server response
            board = data.board;
            score = data.score;
            isGameOver = data.game_over;
            
            // Update UI with new game state
            updateUI();
            
            // Initialize highest tile for the new game
            let highestValue = 0;
            for (let row = 0; row < 4; row++) {
                for (let col = 0; col < 4; col++) {
                    if (data.board[row][col] > 0) {
                        highestValue = Math.max(highestValue, data.board[row][col]);
                    }
                }
            }
            
            if (highestValue > 0) {
                recordHighestTile(highestValue);
            }
            
            // Reset flag after new game is fully loaded
            isResetting = false;
            
            console.log("New game started successfully");
        })
        .catch(error => {
            console.error('Error starting new game:', error);
            isResetting = false;
        });
    }

    /**
     * Stop the AI and clear any pending moves
     */
    function stopAI() {
        console.log("Stopping AI");
        
        // Set flag to stop AI
        gameRunning = false;
        
        // Update button text
        resumeButton.textContent = 'Start AI';
        
        // Clear the timeout if there's a pending AI move
        if (aiTimeoutId !== null) {
            clearTimeout(aiTimeoutId);
            aiTimeoutId = null;
        }
        
        // Cancel any in-flight fetch requests
        if (currentAbortController !== null) {
            currentAbortController.abort();
            currentAbortController = null;
        }
    }

    /**
     * Update the UI to reflect the current game state
     */
    function updateUI() {
        // Clear existing tiles
        const tiles = document.querySelectorAll('.tile');
        tiles.forEach(tile => tile.remove());
        
        // Create new tiles based on the board state
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                if (board[row][col] > 0) {
                    createTile(row, col, board[row][col]);
                }
            }
        }
        
        // Update score
        scoreDisplay.textContent = score;
        
        // Update best score if needed
        if (score > bestScore) {
            bestScore = score;
            bestDisplay.textContent = bestScore;
            localStorage.setItem('bestScore', bestScore);
        } else {
            bestDisplay.textContent = bestScore;
        }
        
        // Check if game is over
        if (isGameOver) {
            showGameOver();
        }
    }

    /**
     * Create a new tile at the specified position
     */
    function createTile(row, col, value, isNew = false, isMerged = false) {
        const tile = document.createElement('div');
        tile.className = `tile tile-${value}${isNew ? ' tile-new' : ''}${isMerged ? ' tile-merged' : ''}`;
        tile.textContent = value;
        tile.style.left = `${col * 116.25 + 10}px`;
        tile.style.top = `${row * 116.25 + 10}px`;
        gridContainer.appendChild(tile);
        return tile;
    }

    /**
     * Handle key press events for manual play
     */
    function handleKeyPress(event) {
        // Ignore key presses if game is over or AI is running
        if (isGameOver || gameRunning) return;
        
        let direction = null;

        aiSpeed = 'medium'
        updateSpeedSelection();
        updateAnimationSpeeds();
        
        switch (event.key) {
            case 'ArrowLeft':
                direction = DIRECTIONS.LEFT;
                break;
            case 'ArrowUp':
                direction = DIRECTIONS.UP;
                break;
            case 'ArrowRight':
                direction = DIRECTIONS.RIGHT;
                break;
            case 'ArrowDown':
                direction = DIRECTIONS.DOWN;
                break;
            default:
                return; // Ignore other keys
        }
        
        // Prevent default behavior (scrolling)
        event.preventDefault();
        
        // Make the move
        makeMove(direction);
    }

    /**
     * Make a move in the specified direction
     */
    function makeMove(direction) {
        fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                game_id: gameId,
                direction: direction
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.moved) {
                // If there's a new highest tile, record it immediately
                // This ensures the display doesn't flicker during animations
                if (data.highest_tile) {
                    recordHighestTile(data.highest_tile);
                }
                
                // Store current display state to restore if needed
                const currentHighestTilesHtml = highestTilesDisplay.innerHTML;
                
                // Animate tiles
                animateMove(data);
                
                // Make sure the highest tiles display wasn't cleared during animation
                if (highestTilesDisplay.innerHTML === '') {
                    highestTilesDisplay.innerHTML = currentHighestTilesHtml;
                }
                
                // Get the computed animation durations for scheduling
                const computedStyle = getComputedStyle(document.documentElement);
                const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
                const mergeDelay = parseInt(computedStyle.getPropertyValue('--merge-delay')) || 50;
                const totalAnimDuration = movementDuration + mergeDelay + 100;
                
                // Update other game state after animations
                setTimeout(() => {
                    // Check if game is being reset
                    if (isResetting) {
                        console.log("Animation complete but game is resetting, ignoring updates");
                        return;
                    }
                    
                    board = data.board;
                    score = data.score;
                    isGameOver = data.game_over;
                    
                    // Make one final check that highest tiles display is still visible
                    if (highestTilesDisplay.innerHTML === '') {
                        updateHighestTilesDisplay();
                    }
                    
                    if (isGameOver) {
                        showGameOver();
                    }
                }, totalAnimDuration);
            }
        })
        .catch(error => console.error('Error making move:', error));
    }

    /**
     * Animate tile movements with proper sliding for merged tiles
     */
    function animateMove(data) {
        // We need both old and new board states
        const oldBoard = board; // Current board state (before update)
        const newBoard = data.board; // New board state from server
        
        // Track the specific new tile position
        let newTileRow = -1;
        let newTileCol = -1;
        
        if (data.new_tile) {
            newTileRow = data.new_tile.position[0];
            newTileCol = data.new_tile.position[1];
        }
        
        // If we have movement data, use it to track which tiles merged
        const mergeMap = new Map(); // Maps destination position to source positions
        const movedTiles = new Map(); // Maps destination position to source position
        
        if (data.movements && data.merges) {
            // First, track all movements
            data.movements.forEach(move => {
                const fromKey = `${move.from[0]},${move.from[1]}`;
                const toKey = `${move.to[0]},${move.to[1]}`;
                movedTiles.set(toKey, fromKey);
            });
            
            // Then, identify merges and their sources
            data.merges.forEach(merge => {
                const pos = `${merge.position[0]},${merge.position[1]}`;
                const sourceTiles = [];
                
                // Find all movements that end at this merge position
                for (const [toPos, fromPos] of movedTiles.entries()) {
                    if (toPos === pos) {
                        sourceTiles.push(fromPos.split(',').map(Number));
                    }
                }
                
                // Store source positions for this merge
                if (sourceTiles.length > 0) {
                    mergeMap.set(pos, sourceTiles);
                }
            });
        }
        
        // Remove all tiles from the DOM
        const tiles = document.querySelectorAll('.tile');
        tiles.forEach(tile => tile.remove());
        
        // Keep track of all tiles that need animation
        const tilesToAnimate = [];
        
        // Process regular moves and merges
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const newValue = newBoard[row][col];
                const posKey = `${row},${col}`;
                
                // Skip empty cells
                if (newValue === 0) continue;
                
                // Skip the new random tile - we'll add it separately
                if (row === newTileRow && col === newTileCol) continue;
                
                // Check if this position has a merged tile
                if (data.merges && data.merges.some(m => m.position[0] === row && m.position[1] === col)) {
                    // This is a merge position
                    const sourceTiles = mergeMap.get(posKey) || [];
                    
                    if (sourceTiles.length > 0) {
                        // We know the exact source tiles
                        
                        // For animation, we'll create a tile at each source position
                        for (const [sourceRow, sourceCol] of sourceTiles) {
                            // Create a tile at the source position
                            const tile = document.createElement('div');
                            tile.className = `tile tile-${newValue/2}`; // Use the pre-merged value
                            tile.textContent = newValue/2;
                            tile.style.left = `${sourceCol * 116.25 + 10}px`;
                            tile.style.top = `${sourceRow * 116.25 + 10}px`;
                            gridContainer.appendChild(tile);
                            
                            // Add to animation list
                            tilesToAnimate.push({
                                tile: tile,
                                fromRow: sourceRow,
                                fromCol: sourceCol,
                                toRow: row,
                                toCol: col,
                                value: newValue/2,
                                finalValue: newValue,
                                isMerge: true
                            });
                        }
                    } else {
                        // Fallback - create tile directly at merge position
                        const tile = document.createElement('div');
                        tile.className = `tile tile-${newValue}`;
                        tile.textContent = newValue;
                        tile.style.left = `${col * 116.25 + 10}px`;
                        tile.style.top = `${row * 116.25 + 10}px`;
                        gridContainer.appendChild(tile);
                        
                        // Add merge animation
                        setTimeout(() => {
                            tile.classList.add('tile-merged');
                        }, parseInt(getComputedStyle(document.documentElement).getPropertyValue('--movement-duration')) || 200);
                    }
                } else if (data.movements) {
                    // Check if this is a moved tile
                    const sourceKey = movedTiles.get(posKey);
                    
                    if (sourceKey) {
                        // This tile was moved from another position
                        const [sourceRow, sourceCol] = sourceKey.split(',').map(Number);
                        
                        // Create tile at source position
                        const tile = document.createElement('div');
                        tile.className = `tile tile-${newValue}`;
                        tile.textContent = newValue;
                        tile.style.left = `${sourceCol * 116.25 + 10}px`;
                        tile.style.top = `${sourceRow * 116.25 + 10}px`;
                        gridContainer.appendChild(tile);
                        
                        // Add to animation list
                        tilesToAnimate.push({
                            tile: tile,
                            fromRow: sourceRow,
                            fromCol: sourceCol,
                            toRow: row,
                            toCol: col,
                            value: newValue,
                            finalValue: newValue,
                            isMerge: false
                        });
                    } else {
                        // Static tile (was already here)
                        const tile = document.createElement('div');
                        tile.className = `tile tile-${newValue}`;
                        tile.textContent = newValue;
                        tile.style.left = `${col * 116.25 + 10}px`;
                        tile.style.top = `${row * 116.25 + 10}px`;
                        gridContainer.appendChild(tile);
                    }
                } else {
                    // Without movement data, place tile directly
                    const tile = document.createElement('div');
                    tile.className = `tile tile-${newValue}`;
                    tile.textContent = newValue;
                    tile.style.left = `${col * 116.25 + 10}px`;
                    tile.style.top = `${row * 116.25 + 10}px`;
                    gridContainer.appendChild(tile);
                }
            }
        }
        
        // Ensure the DOM has time to render the tiles in their initial positions
        // before starting animations
        requestAnimationFrame(() => {
            // Get computed animation durations
            const computedStyle = getComputedStyle(document.documentElement);
            const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
            const mergeDelay = parseInt(computedStyle.getPropertyValue('--merge-delay')) || 50;
            
            // Start animations
            for (const animInfo of tilesToAnimate) {
                const { tile, fromRow, fromCol, toRow, toCol, value, finalValue, isMerge } = animInfo;
                
                // Add transition using CSS variables
                tile.style.transition = `top var(--movement-duration) ease, left var(--movement-duration) ease`;
                
                // Force a reflow to ensure the transition is applied
                tile.offsetHeight;
                
                // Move to destination
                tile.style.top = `${toRow * 116.25 + 10}px`;
                tile.style.left = `${toCol * 116.25 + 10}px`;
                
                // If it's a merge, handle the merge after movement completes
                if (isMerge) {
                    setTimeout(() => {
                        // Remove the original tiles
                        tile.remove();
                        
                        // Check if we need to create the merged tile
                        // (we only need to do this once per merge position)
                        const existingMergedTile = document.querySelector(
                            `.tile-merged[style*="top: ${toRow * 116.25 + 10}px"][style*="left: ${toCol * 116.25 + 10}px"]`
                        );
                        
                        if (!existingMergedTile) {
                            // Create the merged tile
                            const mergedTile = document.createElement('div');
                            mergedTile.className = `tile tile-${finalValue} tile-merged`;
                            mergedTile.textContent = finalValue;
                            mergedTile.style.top = `${toRow * 116.25 + 10}px`;
                            mergedTile.style.left = `${toCol * 116.25 + 10}px`;
                            gridContainer.appendChild(mergedTile);
                        }
                    }, movementDuration + mergeDelay);
                }
            }
        });
        
        // Add the new tile with proper animation - much sooner
        if (data.new_tile) {
            // Get computed animation duration
            const computedStyle = getComputedStyle(document.documentElement);
            const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
            
            // Create the new tile much earlier - right after the movement starts
            setTimeout(() => {
                const row = newTileRow;
                const col = newTileCol;
                const value = data.new_tile.value;
                
                // Create new tile with 'new' animation
                const newTile = document.createElement('div');
                newTile.className = `tile tile-${value} tile-new`;
                newTile.textContent = value;
                newTile.style.left = `${col * 116.25 + 10}px`;
                newTile.style.top = `${row * 116.25 + 10}px`;
                gridContainer.appendChild(newTile);
            }, movementDuration); // Only wait for movement animation, don't add the other delays
        }
        
        // Update score display
        if (data.score !== undefined) {
            scoreDisplay.textContent = data.score;
            if (data.score > bestScore) {
                bestScore = data.score;
                bestDisplay.textContent = bestScore;
                localStorage.setItem('bestScore', bestScore);
            }
        }
    }

    /**
     * Toggle AI gameplay
     */
    function toggleAI() {
        if (gameRunning) {
            stopAI();
        } else {
            // Update speed selection before starting AI
            updateSpeedSelection();
            
            // Start AI
            gameRunning = true;
            resumeButton.textContent = 'Stop AI';
            runAI();
        }
    }

    /**
     * Run the AI gameplay loop
     */
    function runAI() {
        // Check if game is over, AI is stopped, or game is resetting
        if (!gameRunning || isGameOver || isResetting) {
            console.log("AI stopped: gameRunning=", gameRunning, "isGameOver=", isGameOver, "isResetting=", isResetting);
            stopAI();
            return;
        }
        
        console.log("Making AI move");
        
        // Create a new AbortController for this request
        currentAbortController = new AbortController();
        
        fetch('/api/ai_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                game_id: gameId,
                ai_type: aiAgent
            }),
            signal: currentAbortController.signal // Add the abort signal
        })
        .then(response => response.json())
        .then(data => {
            // Check again if game is still running or not resetting
            if (!gameRunning || isResetting) {
                console.log("AI stopped during fetch: gameRunning=", gameRunning, "isResetting=", isResetting);
                return;
            }
            
            if (data.moved) {
                // Record highest tile immediately before animations
                if (data.highest_tile) {
                    recordHighestTile(data.highest_tile);
                }
                
                // Store current display state to restore if needed
                const currentHighestTilesHtml = highestTilesDisplay.innerHTML;
                
                // Animate tiles
                animateMove(data);
                
                // Make sure the highest tiles display wasn't cleared during animation
                if (highestTilesDisplay.innerHTML === '') {
                    highestTilesDisplay.innerHTML = currentHighestTilesHtml;
                }
                
                // Update game state
                board = data.board;
                score = data.score;
                isGameOver = data.game_over;
                
                // Update agent stats if available
                if (data.agent_stats) {
                    updateAgentStats(data.agent_stats);
                }
                
                // Calculate timing for next move based on current animation speed
                const animationDuration = updateAnimationSpeeds();
                
                // Make one final check that highest tiles display is still visible
                if (highestTilesDisplay.innerHTML === '') {
                    updateHighestTilesDisplay();
                }
                
                // Continue AI if game is not over and not resetting
                if (!isGameOver && gameRunning && !isResetting) {
                    // Schedule next move based on speed, but ensuring animations complete first
                    const nextMoveDelay = Math.max(SPEEDS[aiSpeed], animationDuration);
                    
                    aiTimeoutId = setTimeout(() => {
                        if (!isResetting) {
                            runAI();
                        }
                    }, nextMoveDelay);
                } else if (isGameOver) {
                    showGameOver();
                    stopAI();
                }
            } else if (data.game_over) {
                // Game is over without moving
                isGameOver = true;
                showGameOver();
                stopAI();
            } else {
                // No valid move but game not over
                stopAI();
            }
        })
        .catch(error => {
            // Check if this was an abort error (which is expected when stopping)
            if (error.name === 'AbortError') {
                console.log('AI move request was aborted');
            } else {
                console.error('Error getting AI move:', error);
                stopAI();
            }
        });
    }

    /**
     * Update the AI agent type and restart the game
     */
    function updateAIAgent() {
        for (const radio of aiAgentsRadios) {
            if (radio.checked) {
                aiAgent = radio.value;
                break;
            }
        }
        
        // Reset agent stats display
        agentStats.innerHTML = '<p>No stats available yet</p>';
        
        // Restart the game with the new AI
        startNewGame();
    }

    /**
     * Update the AI speed
     */
    function updateAISpeed() {
        // Get the selected speed
        for (const radio of speedRadios) {
            if (radio.checked) {
                aiSpeed = radio.value;
                break;
            }
        }
        
        // Update animation speeds based on new AI speed
        updateAnimationSpeeds();
    }

    /**
     * Ensure the correct speed button is selected
     */
    function updateSpeedSelection() {
        // Find the radio button matching the current aiSpeed and check it
        for (const radio of speedRadios) {
            if (radio.value === aiSpeed) {
                radio.checked = true;
                break;
            }
        }
    }

    /**
     * Update the agent stats display
     */
    function updateAgentStats(stats) {
        if (!stats || Object.keys(stats).length === 0) {
            agentStats.innerHTML = '<p>No stats available</p>';
            return;
        }
        
        let statsHtml = '';
        for (const [key, value] of Object.entries(stats)) {
            // Format the key by replacing underscores with spaces and capitalizing
            const formattedKey = key.replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
                
            statsHtml += `<p><span class="stat-label">${formattedKey}:</span> ${value}</p>`;
        }
        
        agentStats.innerHTML = statsHtml;
    }

    /**
     * Record the highest tile achieved with animation for new high tiles
     */
    function recordHighestTile(value) {
        if (!value || value <= 0) return;
        
        // Check if this is a new high value we haven't seen before
        const isNewHighValue = !tileHistory[value];
        
        // Track this value in history
        tileHistory[value] = true;
        
        // Update display
        updateHighestTilesDisplay(isNewHighValue ? value : null);
    }

    /**
     * Update the highest tiles display - show on a single row, left-aligned
     * @param {number|null} animatedValue - The value to animate with pop effect (if any)
     */
    function updateHighestTilesDisplay(animatedValue = null) {
        // Clear the display first
        highestTilesDisplay.innerHTML = '';
        
        // If we have no tile history, nothing to display
        if (Object.keys(tileHistory).length === 0) {
            return;
        }
        
        // Get all tile values and sort them
        const tileValues = Object.keys(tileHistory)
            .map(val => parseInt(val))
            .sort((a, b) => b - a);
        
        // Create a container for the row with a specific class
        const rowContainer = document.createElement('div');
        rowContainer.className = 'highest-tiles-row';
        rowContainer.style.display = 'flex';
        rowContainer.style.flexDirection = 'row';
        rowContainer.style.justifyContent = 'flex-start'; // Left alignment
        rowContainer.style.flexWrap = 'nowrap';
        rowContainer.style.width = '100%';
        
        // Take only top 6 values
        const topValues = tileValues.slice(0, 6);
        
        // Create a tile for each value
        for (const value of topValues) {
            const tile = document.createElement('div');
            
            // Determine if this tile should have the pop animation
            const shouldAnimate = value === animatedValue;
            
            // Apply the appropriate classes
            if (shouldAnimate) {
                tile.className = `highest-tile tile-${value} tile-merged`;
            } else {
                tile.className = `highest-tile tile-${value}`;
            }
            
            // Adjust styling to ensure they fit on one row
            tile.style.width = '50px';
            tile.style.height = '50px';
            tile.style.fontSize = value >= 1024 ? '16px' : '22px';
            tile.style.position = 'relative';
            tile.style.margin = '0 5px 0 0';
            tile.style.display = 'flex';
            tile.style.justifyContent = 'center';
            tile.style.alignItems = 'center';
            tile.style.borderRadius = '3px';
            tile.style.fontWeight = 'bold';
            
            tile.textContent = value;
            rowContainer.appendChild(tile);
        }
        
        highestTilesDisplay.appendChild(rowContainer);
    }

    /**
     * Show the game over modal
     */
    function showGameOver() {
        // Build game over stats
        const highestTile = Math.max(...board.flat());
        
        let statsHtml = `
            <p>Final Score: ${score}</p>
            <p>Highest Tile: ${highestTile}</p>
        `;
        
        if (aiAgent !== 'human') {
            statsHtml += `<p>AI Agent: ${aiAgent.charAt(0).toUpperCase() + aiAgent.slice(1)}</p>`;
        }
        
        gameOverStats.innerHTML = statsHtml;
        
        // Show the modal
        gameOverModal.style.display = 'flex';
    }
});
