import pygame

# Handles pygame events and user input
class EventHandler:
    def __init__(self):
        self.should_quit = False
        self.paused = False
        self.pause_toggled = False
    
    # Process pygame events. Returns (should_quit, pause_toggled)
    def process_events(self):
        self.pause_toggled = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.should_quit = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.pause_toggled = True
                    print("Training paused" if self.paused else "Training resumed")
        
        # Check if display is still valid
        if not pygame.get_init() or pygame.display.get_surface() is None:
            print("Display closed, saving and exiting...")
            self.should_quit = True
        
        return self.should_quit, self.pause_toggled
    
    def handle_pause(self):
        if self.paused:
            pygame.time.wait(100)
            return True
        return False