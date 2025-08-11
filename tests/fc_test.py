from openai import OpenAI
import time
import json
import os
from io import BytesIO
import base64

from v2_rl_walk_mujoco import RLWalk
from threading import Thread
import cv2
from mini_bdx_runtime.camera import Cam

# TODO mission : find an object ?


# Your Tools class
class Tools:
    def __init__(self):
        self.cam = Cam()
        self.rl_walk = RLWalk(
            "/home/bdxv2/BEST_WALK_ONNX_2.onnx",
            cutoff_frequency=40,
        )

        Thread(target=self.rl_walk.run, daemon=True).start()

    # def upload_image(self, image_path: str):
    #     image_name = os.path.basename(image_path)
    #     im = cv2.imread(image_path)
    #     im = cv2.resize(im, (512, 512))
    #     cv2.imwrite(image_path, im)
    #     # command = (
    #     #     f"scp {image_path} apirrone@s-nguyen.net:/home/apirrone/webserv/images/"
    #     # )
    #     command = (
    #         f"scp {image_path} apirrone@192.168.10.103:/home/apirrone/webserv/images/"
    #     )
    #     print(command)
    #     url = f"http://s-nguyen.net:4444/images/{image_name}"
    #     os.system(command)
    #     return url


    def move_forward(self, seconds=2):
        seconds = max(2, min(seconds, 5))
        print(f"Moving forward for {seconds} seconds")
        self.rl_walk.last_commands[0] = 0.15
        time.sleep(seconds)
        self.rl_walk.last_commands[0] = 0.0
        print("Stopped moving forward")
        return f"Moved forward for {seconds} seconds successfully"

    def turn_left(self, seconds=2):
        seconds = max(2, min(seconds, 5))
        print(f"Turning left for {seconds} seconds")
        self.rl_walk.last_commands[2] = 1.0
        time.sleep(seconds)
        self.rl_walk.last_commands[2] = 0.0
        print("Stopped turning left")
        return f"Turned left for {seconds} seconds successfully"

    def turn_right(self, seconds=2):
        seconds = max(2, min(seconds, 5))
        print(f"Turning right for {seconds} seconds")
        self.rl_walk.last_commands[2] = -1.0
        time.sleep(seconds)
        self.rl_walk.last_commands[2] = 0.0
        print("Stopped turning right")
        return f"Turned right for {seconds} seconds successfully"

    def move_backward(self, seconds=2):
        seconds = max(2, min(seconds, 5))
        print(f"Moving backward for {seconds} seconds")
        self.rl_walk.last_commands[0] = -0.15
        time.sleep(seconds)
        self.rl_walk.last_commands[0] = 0.0
        print("Stopped moving backward")
        return f"Moved backward for {seconds} seconds successfully"

    def take_picture(self):
        # https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/5
        print("Taking a picture...")
        image64 = self.cam.get_encoded_image()
        url = ("data:image/jpeg;base64," + image64,)
        time.sleep(1)
        print("Picture taken")
        return url

    def play_happy_sound(self):
        self.rl_walk.sounds.play_happy()
        return "Played happy sound"

# Tool instance
tools_instance = Tools()

openai_tools = [
    {
        "type": "function",
        "name": "move_forward",
        "description": "Move forward for a number of seconds",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Number of seconds to move forward (min 2, max 5)",
                }
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_backward",
        "description": "Move backward for a number of seconds",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Number of seconds to move backward (min 2, max 5)",
                }
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "turn_left",
        "description": "Turn left on the spot",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Number of seconds to turn left (min 2, max 5)",
                }
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "turn_right",
        "description": "Turn right on the spot",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Number of seconds to turn right (min 2, max 5)",
                }
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "take_picture",
        "description": "Take a picture",
        "parameters": {
            "type": "object",
            "properties": {},
            # No required properties for taking a picture
        },
    },
    {
        "type": "function",
        "name": "play_happy_sound",
        "description": "Play a happy sound",
        "parameters": {
            "type": "object",
            "properties": {},
            # No required properties for playing a sound
        },
    }
]

# Mapping function names to actual methods
function_map = {
    "move_forward": tools_instance.move_forward,
    "move_backward": tools_instance.move_backward,
    "turn_left": tools_instance.turn_left,
    "turn_right": tools_instance.turn_right,
    "take_picture": tools_instance.take_picture,
}

messages = [
    {
        "role": "system",
        "content": (
            "You are a cute little biped robot that can move around using the following tools: "
            "`move_forward`, `move_backward`, `turn_left`, `turn_right`, 'play_happy_sound' and 'take_picture'. "
            "moving forward for 1 second will make you move forward by about 15 centimeters"
            "turning for 1 second will make you turn about 45 degrees"
            "You can only perform one action at a time. If multiple actions are needed, call them step by step."
            "Explain what you are doing along the way"
            "Always start by taking a picture of the environment so you can see where you are. "
            "When you take a picture, describe what you see in the image. "
            "make sure not to hit any walls or objects. Take pictures regularly to know where you are."
            "Maybe it's a good idea to turn 360 degrees to check all directions. (no need if you already found it)"
            "When given a goal to find something, if you found it, navigate to be in front of it, facing it. You want to be about 1 meter close to it"
            "When you are 1 meter close to the object, play the happy sound"
            ""
        ),
    },
    # {
    #     "role": "user",
    #     "content": "Find the yellow vaccum cleaner !",
    # },
    {
        "role": "user",
        "content": "Find the waste bin and turn around it. Play the happy sound when you are done",
    },
]


# Mapping function names to actual methods
function_map = {
    "move_forward": tools_instance.move_forward,
    "move_backward": tools_instance.move_backward,
    "turn_left": tools_instance.turn_left,
    "turn_right": tools_instance.turn_right,
    "take_picture": tools_instance.take_picture,
    "play_happy_sound": tools_instance.play_happy_sound,
}


client = OpenAI()


def call_function(name, args):
    if name == "move_forward":
        return function_map[name](args["seconds"])
    elif name == "move_backward":
        return function_map[name](args["seconds"])
    elif name == "turn_left":
        return function_map[name](args["seconds"])
    elif name == "turn_right":
        return function_map[name](args["seconds"])
    elif name == "take_picture":
        return function_map[name]()
    elif name == "play_happy_sound":
        return function_map[name]()
    else:
        raise ValueError(f"Unknown function name: {name}")


while True:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        tools=openai_tools,
    )

    if len(response.output) == 1 and response.output[0].type == "function_call":
        print("Only function call, no text response")
    else:
        try:
            print(response.output[0].content[0].text)
        except:
            print("Error occurred while processing response")
    for tool_call in response.output:
        if tool_call.type != "function_call":
            continue

        name = tool_call.name
        args = json.loads(tool_call.arguments)

        result = call_function(name, args)[0]
        messages.append(tool_call)
        if tool_call.name == "take_picture":
            # result is an image URL

            # Add an optional prompt or let GPT interpret the image
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": result}],
                }
            )

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": "Image taken and provided above.",
                }
            )
        else:

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result),
                }
            )
