# display.py
import pygame
import numpy as np
from .board import Board

# Colors
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242)
}

class GameDisplay:
    def __init__(self, game, width=500, height=600):
        self.game = game
        self.width = width
        self.height = height
        self.cell_size = 100
        self.grid_padding = 15
        self.animations = []  # Store animations
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2048 - AI Edition")
        
        # Load fonts
        self.title_font = pygame.font.SysFont('Arial', 36)
        self.score_font = pygame.font.SysFont('Arial', 24)
        self.tile_fonts = {
            2: pygame.font.SysFont('Arial', 48),
            4: pygame.font.SysFont('Arial', 48),
            8: pygame.font.SysFont('Arial', 48),
            16: pygame.font.SysFont('Arial', 40),
            32: pygame.font.SysFont('Arial', 40),
            64: pygame.font.SysFont('Arial', 40),
            128: pygame.font.SysFont('Arial', 36),
            256: pygame.font.SysFont('Arial', 36),
            512: pygame.font.SysFont('Arial', 36),
            1024: pygame.font.SysFont('Arial', 30),
            2048: pygame.font.SysFont('Arial', 30)
        }
        
    def get_tile_position(self, row, col):
        """Get the pixel position for a tile"""
        x = self.grid_padding + col * (self.cell_size + self.grid_padding)
        y = 100 + self.grid_padding + row * (self.cell_size + self.grid_padding)
        return (x, y)
        
    def add_slide_animation(self, from_row, from_col, to_row, to_col, value):
        """Add a slide animation to the queue"""
        from_pos = self.get_tile_position(from_row, from_col)
        to_pos = self.get_tile_position(to_row, to_col)
        
        self.animations.append({
            'type': 'slide',
            'from_pos': from_pos,
            'to_pos': to_pos,
            'current_pos': from_pos,
            'value': value,
            'progress': 0.0,
            'duration': 10  # Number of frames
        })
        
    def add_merge_animation(self, row, col, value):
        """Add a merge (pop) animation to the queue"""
        pos = self.get_tile_position(row, col)
        
        self.animations.append({
            'type': 'merge',
            'pos': pos,
            'value': value,
            'progress': 0.0,
            'duration': 8  # Number of frames
        })
        
    def add_new_tile_animation(self, row, col, value):
        """Add a new tile animation to the queue"""
        pos = self.get_tile_position(row, col)
        
        self.animations.append({
            'type': 'new',
            'pos': pos,
            'value': value,
            'progress': 0.0,
            'duration': 10  # Number of frames
        })
        
    def update_animations(self):
        """Update all active animations"""
        for anim in self.animations:
            anim['progress'] += 1.0 / anim['duration']
            
            if anim['type'] == 'slide':
                # Update position for slide animation
                start_x, start_y = anim['from_pos']
                end_x, end_y = anim['to_pos']
                progress = min(1.0, anim['progress'])
                
                # Easing function (ease-out)
                t = 1.0 - (1.0 - progress) ** 2
                
                current_x = start_x + (end_x - start_x) * t
                current_y = start_y + (end_y - start_y) * t
                anim['current_pos'] = (current_x, current_y)
        
        # Remove completed animations
        self.animations = [a for a in self.animations if a['progress'] < 1.0]
        
    def draw_board(self):
        """Draw the game board and all tiles"""
        # Clear screen
        self.screen.fill((250, 248, 239))
        
        # Draw title and score
        title_text = self.title_font.render("2048", True, (119, 110, 101))
        self.screen.blit(title_text, (20, 20))
        
        score_text = self.score_font.render(f"Score: {self.game.score}", True, (119, 110, 101))
        self.screen.blit(score_text, (20, 60))
        
        # Draw AI button
        pygame.draw.rect(self.screen, (143, 122, 102), (self.width - 140, 20, 120, 40), border_radius=5)
        ai_text = self.score_font.render("AI Move", True, (249, 246, 242))
        self.screen.blit(ai_text, (self.width - 130, 28))
        
        # Draw grid background
        pygame.draw.rect(self.screen, GRID_COLOR, 
                        (self.grid_padding, 100 + self.grid_padding, 
                         4 * self.cell_size + 5 * self.grid_padding,
                         4 * self.cell_size + 5 * self.grid_padding),
                        border_radius=6)
        
        # Draw empty cells
        for row in range(4):
            for col in range(4):
                x, y = self.get_tile_position(row, col)
                pygame.draw.rect(self.screen, EMPTY_CELL_COLOR, 
                                (x, y, self.cell_size, self.cell_size),
                                border_radius=5)
        
        # Get current board state
        board = self.game.board.grid
        
        # Draw tiles based on board state (excluding those being animated)
        animated_tiles = {}
        for anim in self.animations:
            if anim['type'] == 'slide':
                from_row = (anim['from_pos'][1] - 100 - self.grid_padding) // (self.cell_size + self.grid_padding)
                from_col = (anim['from_pos'][0] - self.grid_padding) // (self.cell_size + self.grid_padding)
                animated_tiles[(from_row, from_col)] = True
        
        for row in range(4):
            for col in range(4):
                if (row, col) in animated_tiles:
                    continue  # Skip tiles that are being animated
                    
                value = board[row][col]
                if value != 0:
                    self.draw_tile(row, col, value)
        
        # Draw animations
        for anim in self.animations:
            if anim['type'] == 'slide':
                # Draw sliding tile
                x, y = anim['current_pos']
                self.draw_tile_at_position(x, y, anim['value'])
            elif anim['type'] == 'merge':
                # Draw merging tile with scale effect
                x, y = anim['pos']
                progress = anim['progress']
                scale = 1.0
                
                # Scale up then down
                if progress < 0.5:
                    scale = 1.0 + 0.2 * (progress / 0.5)  # Grow to 1.2x
                else:
                    scale = 1.2 - 0.2 * ((progress - 0.5) / 0.5)  # Shrink to 1.0x
                
                value = anim['value']
                size = self.cell_size * scale
                offset = (self.cell_size - size) / 2
                
                pygame.draw.rect(self.screen, TILE_COLORS[value], 
                                (x + offset, y + offset, size, size),
                                border_radius=5)
                
                font = self.tile_fonts.get(value, self.tile_fonts[2048])
                text = font.render(str(value), True, TEXT_COLORS.get(value, (255, 255, 255)))
                text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
                self.screen.blit(text, text_rect)
            elif anim['type'] == 'new':
                # Draw new tile with grow effect
                x, y = anim['pos']
                progress = anim['progress']
                scale = progress  # Start at 0 and grow to 1.0
                
                value = anim['value']
                size = self.cell_size * scale
                offset = (self.cell_size - size) / 2
                
                pygame.draw.rect(self.screen, TILE_COLORS[value], 
                                (x + offset, y + offset, size, size),
                                border_radius=5)
                
                font = self.tile_fonts.get(value, self.tile_fonts[2048])
                text = font.render(str(value), True, TEXT_COLORS.get(value, (255, 255, 255)))
                text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
                self.screen.blit(text, text_rect)
        
        # Update display
        pygame.display.flip()
    
    def draw_tile(self, row, col, value):
        """Draw a tile at the specified grid position"""
        x, y = self.get_tile_position(row, col)
        self.draw_tile_at_position(x, y, value)
    
    def draw_tile_at_position(self, x, y, value):
        """Draw a tile at the specified pixel position"""
        pygame.draw.rect(self.screen, TILE_COLORS[value], 
                        (x, y, self.cell_size, self.cell_size),
                        border_radius=5)
        
        font = self.tile_fonts.get(value, self.tile_fonts[2048])
        text = font.render(str(value), True, TEXT_COLORS.get(value, (255, 255, 255)))
        text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
        self.screen.blit(text, text_rect)
    
    def get_ai_button_rect(self):
        """Get the rectangle for the AI move button"""
        return pygame.Rect(self.width - 140, 20, 120, 40)
