import pygame
import sys
import numpy as np
from game.board import Board
from game.game import Game2048
from game.display import GameDisplay

def main():
    # Initialize pygame
    pygame.init()
    
    # Create game instance
    game = Game2048()
    
    # Create display
    display = GameDisplay(game)
    
    # Game loop variables
    running = True
    clock = pygame.time.Clock()
    animating = False
    
    # Main game loop
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Only process keyboard input if we're not currently animating
            if not animating and not game.is_game_over():
                if event.type == pygame.KEYDOWN:
                    direction = None
                    if event.key == pygame.K_UP:
                        direction = 0
                    elif event.key == pygame.K_RIGHT:
                        direction = 1
                    elif event.key == pygame.K_DOWN:
                        direction = 2
                    elif event.key == pygame.K_LEFT:
                        direction = 3
                    
                    # Process move if a direction key was pressed
                    if direction is not None:
                        move_result = game.step(direction)
                        if move_result['moved']:
                            # Add animations for all movements
                            for move in move_result['movements']:
                                from_row, from_col = move['from']
                                to_row, to_col = move['to']
                                display.add_slide_animation(from_row, from_col, to_row, to_col, move['value'])
                            
                            # Add animations for all merges
                            for merge in move_result['merges']:
                                row, col = merge['position']
                                display.add_merge_animation(row, col, merge['value'])
                            
                            # Add animation for new tile
                            if move_result['new_tile']:
                                row, col = move_result['new_tile']['position']
                                value = move_result['new_tile']['value']
                                display.add_new_tile_animation(row, col, value)
                            
                            # Set animating flag if there are animations
                            animating = len(display.animations) > 0
        
        # Update animations if we have any
        if animating:
            display.update_animations()
            animating = len(display.animations) > 0
        
        # Draw the board
        display.draw_board()
        
        # Game over message
        if game.is_game_over():
            # Create a semi-transparent overlay
            overlay = pygame.Surface((display.width, display.height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            display.screen.blit(overlay, (0, 0))
            
            # Draw game over text
            font = pygame.font.SysFont('Arial', 48, bold=True)
            game_over_text = font.render("Game Over!", True, (119, 110, 101))
            text_rect = game_over_text.get_rect(center=(display.width//2, display.height//2 - 40))
            display.screen.blit(game_over_text, text_rect)
            
            # Draw final score
            score_font = pygame.font.SysFont('Arial', 32)
            score_text = score_font.render(f"Final Score: {game.score}", True, (119, 110, 101))
            score_rect = score_text.get_rect(center=(display.width//2, display.height//2 + 20))
            display.screen.blit(score_text, score_rect)
            
            # Draw restart hint
            hint_font = pygame.font.SysFont('Arial', 24)
            hint_text = hint_font.render("Press 'R' to restart", True, (119, 110, 101))
            hint_rect = hint_text.get_rect(center=(display.width//2, display.height//2 + 70))
            display.screen.blit(hint_text, hint_rect)
            
            # Check for restart
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                # Reset game
                game = Game2048()
                display.game = game
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Quit pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
