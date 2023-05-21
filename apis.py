import json
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import TextGenerationPipeline, AutoModelWithLMHead
import torch
import os
import openai
import abc
import time
import urllib
import requests
from requests.structures import CaseInsensitiveDict
import ast
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
from cleverwrap import CleverWrap

class Bot(abc.ABC):
    @abc.abstractclassmethod
    def respond(self, utterance):
        pass

    def interact(self):
        while True:
            UTTERANCE = input("You: ")
            print(f"Bot: {self.respond(UTTERANCE)}")     

    def script(self, questions):
        responses = []
        for question in questions:
            responses.append(self.respond(question))
        return responses

class BlenderBot(Bot):
    '''single turn for now'''
    def __init__(self) -> None:
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    def respond(self, utterance):
        inputs = self.tokenizer([utterance], return_tensors="pt")
        reply_ids = self.model.generate(**inputs, max_new_tokens=1000)
        return self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            

class DialoGPT(Bot):
    '''single turn for now'''
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    def respond(self, utterance):
        new_user_input_ids = self.tokenizer.encode(utterance + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_user_input_ids
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

class Kuki(Bot):
    def __init__(self) -> None:
        self.url = "https://devman.kuki.ai/talk"
        self.key = os.getenv("KUKI_KEY")
        self.keys_idx = 0

    def respond(self, utterance):
        headers = CaseInsensitiveDict()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = f"botkey={self.key}&input={utterance}&client_name=foo"
        resp = requests.post(self.url, headers=headers, data=data)
        content = resp.content
        dict_str = content.decode("UTF-8")
        content = ast.literal_eval(dict_str)["responses"]
        return content

    def script(self, questions):
        utterance = questions[0]
        return self.respond(utterance)

class Tencent(Bot):
    def __init__(self) -> None:
        self.cred = credential.Credential(os.getenv("TENCENT_ID"), os.getenv("TENCENT_KEY"))
        self.client = nlp_client.NlpClient(self.cred, "ap-guangzhou")

    def respond(self, input):
        content = []
        req = models.ChatBotRequest()
        for utterance in input:
            params = {"Query": utterance}
            req.from_json_string(json.dumps(params))
            resp = self.client.ChatBot(req)
            respDict = json.loads(resp.to_json_string())
            content.append(respDict["Reply"])
        return content
    
    def script(self, questions):
        return self.respond(questions)


class QYK(Bot):
    def __init__(self) -> None:
        self.url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg='
    
    def respond(self, input):
        content = []
        for utterance in input:
            html = requests.get(self.url + urllib.parse.quote(utterance))
            content.append(html.json()["content"])
        return content
    
    def script(self, questions): 
        return self.respond(questions)

class CleverBot():
    def __init__(self) -> None:
        self.cw = CleverWrap("CLEVERBOT_KEY")
    
    def script(self, questions):
        return [self.cw.say(questions[0])]

class GPT3():
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_KEY")
    
    def script(self, questions):
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=questions[0],
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"]
            )
        
        ret = response["choices"][0]["text"].replace("\n", "").replace("AI:", "")
        if ret[0] == " ":
            ret = ret[1:]
        return [ret]

if __name__ == "__main__":
    bot = GPT3()
    ans = bot.script(["blalala?"])
    print(ans)

    


    
    

