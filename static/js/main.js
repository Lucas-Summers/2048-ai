document.addEventListener('DOMContentLoaded', () => {
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

    function initGame() {
        createGameGrid();
        updateSpeedSelection();
        updateAnimationSpeeds();
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

    function createGameGrid() {
        gridContainer.innerHTML = '';
        
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

    function startNewGame() {
        console.log("Starting new game...");
        
        // Set resetting flag to prevent further AI moves
        isResetting = true;
        
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
        document.getElementById('grid-container').innerHTML = '';
        
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.style.left = `${col * 116.25 + 10}px`;
                cell.style.top = `${row * 116.25 + 10}px`;
                gridContainer.appendChild(cell);
            }
        }
        
        board = [];
        score = 0;
        isGameOver = false;
        tileHistory = {};
        
        scoreDisplay.textContent = score;
        highestTilesDisplay.innerHTML = '';
        
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_id: gameId })
        })
        .then(response => response.json())
        .then(data => {
            board = data.board;
            score = data.score;
            isGameOver = data.game_over;
            
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
            
            isResetting = false;
            
            console.log("New game started successfully");
        })
        .catch(error => {
            console.error('Error starting new game:', error);
            isResetting = false;
        });
    }

    function stopAI() {
        console.log("Stopping AI");
        
        gameRunning = false;
        resumeButton.textContent = 'Start AI';
        
        if (aiTimeoutId !== null) {
            clearTimeout(aiTimeoutId);
            aiTimeoutId = null;
        }
        
        if (currentAbortController !== null) {
            currentAbortController.abort();
            currentAbortController = null;
        }
    }

    function updateUI() {
        const tiles = document.querySelectorAll('.tile');
        tiles.forEach(tile => tile.remove());
        
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                if (board[row][col] > 0) {
                    createTile(row, col, board[row][col]);
                }
            }
        }
        
        scoreDisplay.textContent = score;
        
        if (score > bestScore) {
            bestScore = score;
            bestDisplay.textContent = bestScore;
            localStorage.setItem('bestScore', bestScore);
        } else {
            bestDisplay.textContent = bestScore;
        }
        
        if (isGameOver) {
            showGameOver();
        }
    }

    function createTile(row, col, value, isNew = false, isMerged = false) {
        const tile = document.createElement('div');
        tile.className = `tile tile-${value}${isNew ? ' tile-new' : ''}${isMerged ? ' tile-merged' : ''}`;
        tile.textContent = value;
        tile.style.left = `${col * 116.25 + 10}px`;
        tile.style.top = `${row * 116.25 + 10}px`;
        gridContainer.appendChild(tile);
        return tile;
    }

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
                return;
        }
        
        event.preventDefault();
        makeMove(direction);
    }

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
                // Record highest tile immediately to prevent flicker during animations
                if (data.highest_tile) {
                    recordHighestTile(data.highest_tile);
                }
                
                const currentHighestTilesHtml = highestTilesDisplay.innerHTML;
                
                animateMove(data);
                
                // Restore highest tiles display if cleared during animation
                if (highestTilesDisplay.innerHTML === '') {
                    highestTilesDisplay.innerHTML = currentHighestTilesHtml;
                }
                
                const computedStyle = getComputedStyle(document.documentElement);
                const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
                const mergeDelay = parseInt(computedStyle.getPropertyValue('--merge-delay')) || 50;
                const totalAnimDuration = movementDuration + mergeDelay + 100;
                
                // Update game state after animations complete
                setTimeout(() => {
                    if (isResetting) {
                        console.log("Animation complete but game is resetting, ignoring updates");
                        return;
                    }
                    
                    board = data.board;
                    score = data.score;
                    isGameOver = data.game_over;
                    
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
        const oldBoard = board; // Current board state (before update)
        const newBoard = data.board; // New board state from server
        
        let newTileRow = -1;
        let newTileCol = -1;
        
        if (data.new_tile) {
            newTileRow = data.new_tile.position[0];
            newTileCol = data.new_tile.position[1];
        }
        
        // Track which tiles merged and moved
        const mergeMap = new Map(); // Maps destination position to source positions
        const movedTiles = new Map(); // Maps destination position to source position
        
        if (data.movements && data.merges) {
            data.movements.forEach(move => {
                const fromKey = `${move.from[0]},${move.from[1]}`;
                const toKey = `${move.to[0]},${move.to[1]}`;
                movedTiles.set(toKey, fromKey);
            });
            
            data.merges.forEach(merge => {
                const pos = `${merge.position[0]},${merge.position[1]}`;
                const sourceTiles = [];
                
                for (const [toPos, fromPos] of movedTiles.entries()) {
                    if (toPos === pos) {
                        sourceTiles.push(fromPos.split(',').map(Number));
                    }
                }
                
                if (sourceTiles.length > 0) {
                    mergeMap.set(pos, sourceTiles);
                }
            });
        }
        
        const tiles = document.querySelectorAll('.tile');
        tiles.forEach(tile => tile.remove());
        
        const tilesToAnimate = [];
        
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const newValue = newBoard[row][col];
                const posKey = `${row},${col}`;
                
                if (newValue === 0) continue;
                
                // Skip the new random tile - handle separately
                if (row === newTileRow && col === newTileCol) continue;
                
                if (data.merges && data.merges.some(m => m.position[0] === row && m.position[1] === col)) {
                    const sourceTiles = mergeMap.get(posKey) || [];
                    
                    if (sourceTiles.length > 0) {
                        // Create a tile at each source position for animation
                        for (const [sourceRow, sourceCol] of sourceTiles) {
                            const tile = document.createElement('div');
                            tile.className = `tile tile-${newValue/2}`; // Use the pre-merged value
                            tile.textContent = newValue/2;
                            tile.style.left = `${sourceCol * 116.25 + 10}px`;
                            tile.style.top = `${sourceRow * 116.25 + 10}px`;
                            gridContainer.appendChild(tile);
                            
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
                        
                        setTimeout(() => {
                            tile.classList.add('tile-merged');
                        }, parseInt(getComputedStyle(document.documentElement).getPropertyValue('--movement-duration')) || 200);
                    }
                } else if (data.movements) {
                    const sourceKey = movedTiles.get(posKey);
                    
                    if (sourceKey) {
                        // This tile was moved from another position
                        const [sourceRow, sourceCol] = sourceKey.split(',').map(Number);
                        
                        const tile = document.createElement('div');
                        tile.className = `tile tile-${newValue}`;
                        tile.textContent = newValue;
                        tile.style.left = `${sourceCol * 116.25 + 10}px`;
                        tile.style.top = `${sourceRow * 116.25 + 10}px`;
                        gridContainer.appendChild(tile);
                        
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
        
        // Ensure DOM renders tiles in initial positions before animating
        requestAnimationFrame(() => {
            const computedStyle = getComputedStyle(document.documentElement);
            const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
            const mergeDelay = parseInt(computedStyle.getPropertyValue('--merge-delay')) || 50;
            
            for (const animInfo of tilesToAnimate) {
                const { tile, fromRow, fromCol, toRow, toCol, value, finalValue, isMerge } = animInfo;
                
                tile.style.transition = `top var(--movement-duration) ease, left var(--movement-duration) ease`;
                
                // Force reflow to ensure transition is applied
                tile.offsetHeight;
                
                tile.style.top = `${toRow * 116.25 + 10}px`;
                tile.style.left = `${toCol * 116.25 + 10}px`;
                
                if (isMerge) {
                    setTimeout(() => {
                        tile.remove();
                        
                        // Create merged tile only once per merge position
                        const existingMergedTile = document.querySelector(
                            `.tile-merged[style*="top: ${toRow * 116.25 + 10}px"][style*="left: ${toCol * 116.25 + 10}px"]`
                        );
                        
                        if (!existingMergedTile) {
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
        
        // Add new tile after movement animation completes
        if (data.new_tile) {
            const computedStyle = getComputedStyle(document.documentElement);
            const movementDuration = parseInt(computedStyle.getPropertyValue('--movement-duration')) || 200;
            
            setTimeout(() => {
                const row = newTileRow;
                const col = newTileCol;
                const value = data.new_tile.value;
                
                const newTile = document.createElement('div');
                newTile.className = `tile tile-${value} tile-new`;
                newTile.textContent = value;
                newTile.style.left = `${col * 116.25 + 10}px`;
                newTile.style.top = `${row * 116.25 + 10}px`;
                gridContainer.appendChild(newTile);
            }, movementDuration);
        }
        
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
            updateSpeedSelection();
            
            gameRunning = true;
            resumeButton.textContent = 'Stop AI';
            runAI();
        }
    }

    /**
     * Run the AI gameplay loop
     */
    function runAI() {
        if (!gameRunning || isGameOver || isResetting) {
            console.log("AI stopped: gameRunning=", gameRunning, "isGameOver=", isGameOver, "isResetting=", isResetting);
            stopAI();
            return;
        }
        
        console.log("Making AI move");
        
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
            signal: currentAbortController.signal
        })
        .then(response => response.json())
        .then(data => {
            if (!gameRunning || isResetting) {
                console.log("AI stopped during fetch: gameRunning=", gameRunning, "isResetting=", isResetting);
                return;
            }
            
            if (data.moved) {
                // Record highest tile immediately before animations
                if (data.highest_tile) {
                    recordHighestTile(data.highest_tile);
                }
                
                const currentHighestTilesHtml = highestTilesDisplay.innerHTML;
                
                animateMove(data);
                
                // Restore display if cleared during animation
                if (highestTilesDisplay.innerHTML === '') {
                    highestTilesDisplay.innerHTML = currentHighestTilesHtml;
                }
                
                board = data.board;
                score = data.score;
                isGameOver = data.game_over;
                
                if (data.agent_stats) {
                    updateAgentStats(data.agent_stats);
                }
                
                // Calculate timing for next move based on current animation speed
                const animationDuration = updateAnimationSpeeds();
                
                if (highestTilesDisplay.innerHTML === '') {
                    updateHighestTilesDisplay();
                }
                
                if (!isGameOver && gameRunning && !isResetting) {
                    // Schedule next move, ensuring animations complete first
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
                isGameOver = true;
                showGameOver();
                stopAI();
            } else {
                stopAI();
            }
        })
        .catch(error => {
            // Check if this was an abort error (expected when stopping)
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
        
        agentStats.innerHTML = '<p>No stats available yet</p>';
        startNewGame();
    }

    /**
     * Update the AI speed
     */
    function updateAISpeed() {
        for (const radio of speedRadios) {
            if (radio.checked) {
                aiSpeed = radio.value;
                break;
            }
        }
        
        updateAnimationSpeeds();
    }

    /**
     * Ensure the correct speed button is selected
     */
    function updateSpeedSelection() {
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
        
        // Sort stats - priority stats first, then alphabetical
        const priorityStats = ["thinking_time", "iterations", "nodes_explored", "max_depth"];
        const sortedKeys = Object.keys(stats).sort((a, b) => {
            const aIndex = priorityStats.indexOf(a);
            const bIndex = priorityStats.indexOf(b);
            
            if (aIndex !== -1 && bIndex !== -1) {
                return aIndex - bIndex;
            }
            
            if (aIndex !== -1) return -1;
            if (bIndex !== -1) return 1;
            
            return a.localeCompare(b);
        });
        
        for (const key of sortedKeys) {
            const value = stats[key];
            
            const formattedKey = key.replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
            
            let formattedValue = value;
            if (typeof value === 'number' && !Number.isInteger(value)) {
                formattedValue = value.toFixed(2);
            }
            
            statsHtml += `<p><span class="stat-label">${formattedKey}:</span> ${formattedValue}</p>`;
        }
        
        agentStats.innerHTML = statsHtml;
    }

    /**
     * Record the highest tile achieved with animation for new high tiles
     */
    function recordHighestTile(value) {
        if (!value || value <= 0) return;
        
        const isNewHighValue = !tileHistory[value];
        
        tileHistory[value] = true;
        
        updateHighestTilesDisplay(isNewHighValue ? value : null);
    }

    /**
     * Update the highest tiles display - show on a single row, left-aligned
     * @param {number|null} animatedValue - The value to animate with pop effect (if any)
     */
    function updateHighestTilesDisplay(animatedValue = null) {
        highestTilesDisplay.innerHTML = '';
        
        if (Object.keys(tileHistory).length === 0) {
            return;
        }
        
        const tileValues = Object.keys(tileHistory)
            .map(val => parseInt(val))
            .sort((a, b) => b - a);
        
        const rowContainer = document.createElement('div');
        rowContainer.className = 'highest-tiles-row';
        rowContainer.style.display = 'flex';
        rowContainer.style.flexDirection = 'row';
        rowContainer.style.justifyContent = 'flex-start';
        rowContainer.style.flexWrap = 'nowrap';
        rowContainer.style.width = '100%';
        
        // Take only top 6 values
        const topValues = tileValues.slice(0, 6);
        
        for (const value of topValues) {
            const tile = document.createElement('div');
            
            const shouldAnimate = value === animatedValue;
            
            if (shouldAnimate) {
                tile.className = `highest-tile tile-${value} tile-merged`;
            } else {
                tile.className = `highest-tile tile-${value}`;
            }
            
            // Styling to fit on one row
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
        const highestTile = Math.max(...board.flat());
        
        let statsHtml = `
            <p>Final Score: ${score}</p>
            <p>Highest Tile: ${highestTile}</p>
        `;
        
        if (aiAgent !== 'human') {
            statsHtml += `<p>AI Agent: ${aiAgent.charAt(0).toUpperCase() + aiAgent.slice(1)}</p>`;
        }
        
        gameOverStats.innerHTML = statsHtml;
        gameOverModal.style.display = 'flex';
    }
});
