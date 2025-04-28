document.addEventListener('DOMContentLoaded', function() {
    // Game state
    let gameId = 'default';
    let gameRunning = false;
    let aiInterval = null;
    let bestScore = 0;
    let highestTiles = [];
    
    // DOM elements
    const gridContainer = document.getElementById('grid-container');
    const scoreElement = document.getElementById('score');
    const bestElement = document.getElementById('best');
    const resumeButton = document.getElementById('resume-button');
    const newGameButton = document.getElementById('new-game-button');
    const highestTilesContainer = document.getElementById('highest-tiles');
    
    // Initialize the game
    initGame();
    
    // Setup event listeners
    resumeButton.addEventListener('click', toggleAI);
    newGameButton.addEventListener('click', startNewGame);
    document.addEventListener('keydown', handleKeyPress);
    
    // Setup AI mode radio buttons
    document.querySelectorAll('input[name="ai-mode"]').forEach(radio => {
        radio.addEventListener('change', updateAISettings);
    });
    
    // Setup speed radio buttons
    document.querySelectorAll('input[name="speed"]').forEach(radio => {
        radio.addEventListener('change', updateAISettings);
    });
    
    // Setup tile generator radio buttons
    document.querySelectorAll('input[name="tile-generator"]').forEach(radio => {
        radio.addEventListener('change', updateTileGenerator);
    });
    
    // Initialize the game
    function initGame() {
        // Create grid cells
        gridContainer.innerHTML = '';
        
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                
                // Position the cell
                const x = col * 117;  // 106.25px cell width + gap
                const y = row * 117;  // 106.25px cell height + gap
                cell.style.left = `${x}px`;
                cell.style.top = `${y}px`;
                
                gridContainer.appendChild(cell);
            }
        }
        
        // Start a new game
        startNewGame();
    }
    
    // Start a new game
    function startNewGame() {
        stopAI();
        resumeButton.textContent = 'Resume';
        gameRunning = false;
        
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ game_id: gameId }),
        })
        .then(response => response.json())
        .then(data => {
            updateBoard(data.board);
            updateScore(data.score);
            highestTiles = [];
            updateHighestTiles();
        });
    }
    
    // Update the board display
    function updateBoard(board) {
        // Clear existing tiles
        const existingTiles = document.querySelectorAll('.tile');
        existingTiles.forEach(tile => tile.remove());
        
        // Create new tiles
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                const value = board[row][col];
                if (value > 0) {
                    createTile(row, col, value);
                }
            }
        }
    }
    
    // Create a tile element
    function createTile(row, col, value) {
        const tile = document.createElement('div');
        tile.className = `tile tile-${value}`;
        tile.textContent = value;
        
        // Position the tile
        const x = col * 117;  // 106.25px tile width + gap
        const y = row * 117;  // 106.25px tile height + gap
        tile.style.left = `${x}px`;
        tile.style.top = `${y}px`;
        
        gridContainer.appendChild(tile);
        return tile;
    }
    
    // Update score display
    function updateScore(score) {
        scoreElement.textContent = score;
        
        if (score > bestScore) {
            bestScore = score;
            bestElement.textContent = bestScore;
        }
    }
    
    // Handle keyboard input
    function handleKeyPress(event) {
        if (gameRunning) return; // Ignore keyboard if AI is running
        
        let direction = null;
        
        switch (event.key) {
            case 'ArrowUp':
                direction = 0;
                break;
            case 'ArrowRight':
                direction = 1;
                break;
            case 'ArrowDown':
                direction = 2;
                break;
            case 'ArrowLeft':
                direction = 3;
                break;
        }
        
        if (direction !== null) {
            makeMove(direction);
            event.preventDefault();
        }
    }
    
    // Make a move
    function makeMove(direction) {
        fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game_id: gameId,
                direction: direction
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.moved) {
                // Animate moves
                animateMoves(data.movements, data.merges, data.new_tile);
                
                // Update score
                updateScore(data.score);
                
                // Check game over
                if (data.game_over) {
                    setTimeout(() => {
                        alert('Game Over!');
                    }, 500);  // Delay alert to allow animations to complete
                }
            }
        });
    }
    
    // Animate moves with proper sliding and merging
    function animateMoves(movements, merges, newTile) {
        // Track which animations are in progress
        let animationsInProgress = 0;
        
        // Process movements first
        movements.forEach(move => {
            const fromRow = move.from[0];
            const fromCol = move.from[1];
            const toRow = move.to[0];
            const toCol = move.to[1];
            const value = move.value;
            
            // Create the sliding tile
            const tile = document.createElement('div');
            tile.className = `tile tile-${value}`;
            tile.textContent = value;
            
            // Set starting position
            const startX = fromCol * 117;
            const startY = fromRow * 117;
            tile.style.left = `${startX}px`;
            tile.style.top = `${startY}px`;
            
            // Add to grid
            gridContainer.appendChild(tile);
            
            // Track this animation
            animationsInProgress++;
            
            // Start animation after a brief delay (for browser to process)
            setTimeout(() => {
                // Set ending position
                const endX = toCol * 117;
                const endY = toRow * 117;
                
                // Add transition class
                tile.style.transition = 'left 0.15s ease, top 0.15s ease';
                tile.style.left = `${endX}px`;
                tile.style.top = `${endY}px`;
                
                // Listen for the end of transition
                tile.addEventListener('transitionend', function handler() {
                    // Remove the transition event listener
                    tile.removeEventListener('transitionend', handler);
                    
                    // Check if this tile should merge
                    const shouldMerge = merges.some(merge => 
                        merge.position[0] === toRow && merge.position[1] === toCol);
                    
                    if (!shouldMerge) {
                        // If not merging, just remove the animation tile
                        tile.remove();
                    }
                    
                    // Mark this animation as complete
                    animationsInProgress--;
                    
                    // If all animations are done, process the next steps
                    if (animationsInProgress === 0) {
                        processMergesAndNewTile(merges, newTile);
                    }
                });
            }, 50);
        });
        
        // If there were no movements, still process merges and new tile
        if (movements.length === 0) {
            processMergesAndNewTile(merges, newTile);
        }
    }
    
    // Process merges and new tile animations
    function processMergesAndNewTile(merges, newTile) {
        let animationsInProgress = 0;
        
        // Process merges
        merges.forEach(merge => {
            const row = merge.position[0];
            const col = merge.position[1];
            const value = merge.value;
            
            // Find any existing tiles at this position and remove them
            const existingTiles = Array.from(document.querySelectorAll('.tile')).filter(tile => {
                const left = parseInt(tile.style.left);
                const top = parseInt(tile.style.top);
                const tileCol = Math.round(left / 117);
                const tileRow = Math.round(top / 117);
                return tileRow === row && tileCol === col;
            });
            
            existingTiles.forEach(tile => tile.remove());
            
            // Create the merged tile
            const tile = document.createElement('div');
            tile.className = `tile tile-${value}`;
            tile.textContent = value;
            
            // Position the tile
            const x = col * 117;
            const y = row * 117;
            tile.style.left = `${x}px`;
            tile.style.top = `${y}px`;
            
            // Add to grid
            gridContainer.appendChild(tile);
            
            // Track this animation
            animationsInProgress++;
            
            // Add merge animation
            setTimeout(() => {
                tile.classList.add('tile-merged');
                
                // Listen for animation end
                tile.addEventListener('animationend', function handler() {
                    tile.removeEventListener('animationend', handler);
                    tile.classList.remove('tile-merged');
                    
                    // Mark animation as complete
                    animationsInProgress--;
                    
                    // If all animations are done, add the new tile
                    if (animationsInProgress === 0 && newTile) {
                        addNewTileAnimation(newTile);
                    }
                });
            }, 50);
        });
        
        // If no merges, directly add the new tile
        if (merges.length === 0 && newTile) {
            addNewTileAnimation(newTile);
        }
    }
    
    // Add new tile with animation
    function addNewTileAnimation(newTile) {
        const row = newTile.position[0];
        const col = newTile.position[1];
        const value = newTile.value;
        
        // Create the new tile
        const tile = document.createElement('div');
        tile.className = `tile tile-${value} tile-new`;
        tile.textContent = value;
        
        // Position the tile
        const x = col * 117;
        const y = row * 117;
        tile.style.left = `${x}px`;
        tile.style.top = `${y}px`;
        
        // Add to grid
        gridContainer.appendChild(tile);
        
        // Remove animation class after animation completes
        setTimeout(() => {
            tile.classList.remove('tile-new');
        }, 300);
    }
    
    // Toggle AI on/off
    function toggleAI() {
        if (gameRunning) {
            stopAI();
            resumeButton.textContent = 'Resume';
        } else {
            startAI();
            resumeButton.textContent = 'Pause';
        }
    }
    
    // Start AI
    function startAI() {
        gameRunning = true;
        
        // Get AI speed
        const speedRadios = document.querySelectorAll('input[name="speed"]');
        let speed = 500; // Default medium speed
        
        for (const radio of speedRadios) {
            if (radio.checked) {
                if (radio.value === 'full') speed = 100;
                else if (radio.value === 'fast') speed = 300;
                else if (radio.value === 'slow') speed = 800;
                break;
            }
        }
        
        // Start AI loop
        aiInterval = setInterval(makeAIMove, speed);
    }
    
    // Stop AI
    function stopAI() {
        gameRunning = false;
        if (aiInterval) {
            clearInterval(aiInterval);
            aiInterval = null;
        }
    }
    
    // Make an AI move
    function makeAIMove() {
        // Get the selected AI mode
        const aiModeRadios = document.querySelectorAll('input[name="ai-mode"]');
        let aiType = 'random';
        
        for (const radio of aiModeRadios) {
            if (radio.checked) {
                aiType = radio.value;
                break;
            }
        }
        
        fetch('/api/ai_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game_id: gameId,
                ai_type: aiType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.moved) {
                // Animate moves
                animateMoves(data.movements, data.merges, data.new_tile);
                
                // Update score
                updateScore(data.score);
                
                // Track highest tiles
                updateHighestTilesList(data.highest_tile);
                
                // Check game over
                if (data.game_over) {
                    stopAI();
                    resumeButton.textContent = 'New Game';
                    setTimeout(() => {
                        alert('Game Over! Final Score: ' + data.score);
                    }, 500);  // Delay alert to allow animations to complete
                }
            } else {
                // If AI can't move, stop
                stopAI();
                resumeButton.textContent = 'New Game';
                setTimeout(() => {
                    alert('AI can\'t move anymore!');
                }, 500);
            }
        });
    }
    
    // Update AI settings
    function updateAISettings() {
        // If AI is running, restart it with new settings
        if (gameRunning) {
            stopAI();
            startAI();
        }
    }
    
    // Update tile generator settings
    function updateTileGenerator() {
        // Send tile generator settings to backend
        const tileGeneratorRadios = document.querySelectorAll('input[name="tile-generator"]');
        let generator = 'random';
        
        for (const radio of tileGeneratorRadios) {
            if (radio.checked) {
                generator = radio.value;
                break;
            }
        }
        
        fetch('/api/update_generator', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game_id: gameId,
                generator: generator
            }),
        });
    }
    
    // Update highest tiles list
    function updateHighestTilesList(highestTile) {
        if (highestTile && !highestTiles.includes(highestTile)) {
            if (highestTiles.length >= 8) {
                highestTiles.shift();
            }
            highestTiles.push(highestTile);
            updateHighestTiles();
        }
    }
    
    // Update highest tiles display
    function updateHighestTiles() {
        highestTilesContainer.innerHTML = '';
        
        // Sort tiles in descending order
        const sortedTiles = [...highestTiles].sort((a, b) => b - a);
        
        // Display unique tiles
        const uniqueTiles = [...new Set(sortedTiles)];
        uniqueTiles.forEach(value => {
            const tile = document.createElement('div');
            tile.className = `tile tile-${value}`;
            tile.textContent = value;
            tile.style.width = '50px';
            tile.style.height = '50px';
            tile.style.fontSize = value >= 1000 ? '18px' : '24px';
            tile.style.lineHeight = '50px';
            tile.style.position = 'relative';
            tile.style.display = 'inline-block';
            tile.style.margin = '2px';
            
            highestTilesContainer.appendChild(tile);
        });
    }
});
