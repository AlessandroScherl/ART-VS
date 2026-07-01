"""
Centralized model configuration for the ViT-VS project.
Models 4, 5, 7, 11, 12 have been removed due to issues.
"""

# Model mapping - NEW MODELS ADDED (4, 5, 7, 11, 12, 21)
MODELS = {
    1: "1_Coffee_mug_1",
    2: "1_Coffeemaker",
    3: "1_Cooking_pan",
    4: "1_Toaster",  # NEW - Kitchen appliance
    5: "2_Helmet",  # NEW - Safety equipment
    6: "2_Hammer_black",
    7: "2_Scissors",  # NEW - Tool
    8: "2_Screwdriver",
    9: "3_Keyboard",
    10: "3_Laptop_1",
    11: "3_Alarmclock",  # NEW - Electronics
    12: "3_Mouse",  # NEW - Computer accessory
    13: "4_Shoe_boat",
    14: "4_Shoe_boot",
    15: "4_Shoe_sandal",
    16: "4_Shoe_sport",
    17: "5_Sonny_School_Bus",
    18: "5_Toy_squirrel",
    19: "5_Toy_transformer",
    20: "5_Toy_turtle",
    21: "4_Backpack",  # NEW - Accessory
    99: "viso"  # Hollywood experiment
}

# YOLOWorld detection keywords for each model
# Updated with optimized keywords from testing:
# - Model 8: "tool screwdriver" (+20% detection)
# - Model 17: "toy bus" (+30% detection)  
# - Model 19: "action toy" (+50% detection)
YOLO_KEYWORDS = {
    1: "cup",
    2: "coffee maker",
    3: "skillet",
    4: "toaster",  # NEW
    5: "helmet",  # NEW
    6: "tool hammer",
    7: "scissors",  # NEW
    8: "tool screwdriver",  # OPTIMIZED: was "screwdriver" (+20%)
    9: "keyboard",
    10: "laptop",
    11: "alarm clock",  # NEW
    12: "computer mouse",  # NEW
    13: "shoe",
    14: "shoe",
    15: "flip flop",
    16: "shoe",
    17: "toy bus",  # OPTIMIZED: was "bus" (+30%)
    18: "stuffed animal",
    19: "action toy",  # OPTIMIZED: was "action figure" (+50%)
    20: "toy turtle",
    21: "bag",  # NEW - Changed from "backpack" for better detection
    99: "poster"
}

# Goal image mapping for each model
GOAL_IMAGES = {
    1: "goalrgb_mug.jpg",
    2: "goalrgb_coffeemaker.jpg",
    3: "goalrgb_pan.jpg",
    4: "goalrgb_toaster.jpg",  # NEW
    5: "goalrgb_helmet.jpg",  # NEW
    6: "goalrgb_hammer.jpg",
    7: "goalrgb_scissors.jpg",  # NEW
    8: "goalrgb_screwdriver.jpg",
    9: "goalrgb_keyboard.jpg",
    10: "goalrgb_laptop.jpg",
    11: "goalrgb_alarmclock.jpg",  # NEW
    12: "goalrgb_mouse.jpg",  # NEW
    13: "goalrgb_shoe_boat.jpg",
    14: "goalrgb_shoe_boot.jpg",
    15: "goalrgb_shoe_sandal.jpg",
    16: "goalrgb_shoe_sneaker.jpg",
    17: "goalrgb_toy_schoolbus.jpg",
    18: "goalrgb_toy_squirrel.jpg",
    19: "goalrgb_toy_transformer.jpg",
    20: "goalrgb_toy_turtle.jpg",
    21: "goalrgb_backpack.jpg",  # NEW
    99: "goalrgb.jpg"  # Hollywood poster
}

# Valid model indices (including new models)
VALID_MODELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 99]