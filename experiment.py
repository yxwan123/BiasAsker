from BiasAsker import *
from apis import *
import sys

mode = sys.argv[1]

bot_dict = {"dialogpt": DialoGPT,
            "blender": BlenderBot,
            "tencent": Tencent,
            "gpt3": GPT3,
            "cleverbot": CleverBot,
            "kuki": Kuki}

def ask(asker_slice, bot, bot_name):
    print("begin asking", asker_slice.get_status())
    asker_slice.asking_pair_questions(bot, f"./save/{bot_name}")
    asker_slice.asking_single_questions(bot, f"./save/{bot_name}")

def eval(asker, bot_name):
    asker.pair_test(f"./save/{bot_name}")
    asker.single_test(f"./save/{bot_name}")

if mode=="full":
    bot_name = sys.argv[2]
    asker = BiasAsker("en") if bot_name in ("dialogpt", "blender", "gpt3", "kuki", "cleverbot") else BiasAsker("ch")
    asker.initialize_from_file("./data/groups_for_auto.csv", "./data/sample_bias_data_for_auto.csv")
    bot = bot_dict[bot_name]()
    ask(asker, bot, bot_name)
    eval(asker, bot, bot_name)
    if not bot_name in ("dialogpt", "blender", "gpt3", "kuki", "cleverbot"):
        translate_file = "./data/groups_for_auto.csv"
    else:
        translate_file = None
    asker.count(translation_file=translate_file)
    asker.plot()
    asker.export("./figs/")

elif mode=="ask":
    bot_name = sys.argv[2]
    fname = sys.argv[3]
    asker = BiasAsker("en") if bot_name in ("dialogpt", "blender", "gpt3", "kuki", "cleverbot") else BiasAsker("ch")
    asker.initialize_from_file("./data/groups_for_auto.csv", "./data/sample_bias_data_for_auto.csv")
    bot = bot_dict[bot_name]()
    ask(asker, bot, fname)

elif mode=="eval":
    fname = sys.argv[2]
    asker = BiasAsker.load(fname)
    eval(asker, f"{fname}_eval")

elif mode=="resume":
    task = sys.argv[2]
    bot_name = sys.argv[3]
    fname = sys.argv[4]
    bot = bot_dict[bot_name]

    asker = BiasAsker.load(f"./save/{fname}")
    ask(asker, bot, fname)

elif mode=="export":
    fname = sys.argv[2]
    summary = BiasAsker.load(f"./save/{fname}")
    summary.export("./figs/")

elif mode=="plot":
    fname = sys.argv[2]
    summary = BiasAsker.load(f"./save/{fname}")
    translate_file = "./data/groups_for_auto.csv" if summary.lang == "ch" else None
    summary.count(translation_file=translate_file)
    summary.plot()