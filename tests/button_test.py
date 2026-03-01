import pygame

pygame.init()

# Initialize the joystick
pygame.joystick.init()

# Check for available joysticks
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected.")
    exit()

# Use the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick Name: {joystick.get_name()}")
print(f"Number of Axes: {joystick.get_numaxes()}")
print(f"Number of Buttons: {joystick.get_numbuttons()}")
print(f"Number of Hats: {joystick.get_numhats()}")

# Main loop to check for button presses
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"Button {event.button} pressed.")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"Button {event.button} released.")
            elif event.type == pygame.JOYAXISMOTION:
                # You can add axis motion event handling here if needed
                pass
            elif event.type == pygame.JOYHATMOTION:
                print(f"Hat {event.hat} value: {event.value}")

except KeyboardInterrupt:
    print("Exiting.")
    pygame.quit()
