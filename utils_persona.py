import copy
import numpy as np
import tiktoken
import torch
import os
import torch
import gc
from datasets import load_dataset
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from datetime import datetime
import re
import numpy as np
from multiprocessing import Process
import pycountry

def create_persona_3basic(x):
    age = int(x['age'])
    gender = int(x['Gender'])
    premium_pupil = x['PremiumPupil']

    if age == 8 or age == 11 or age == 18:
        persona = f"""Pretend that you are an {age}-year-old student. """
    else:
        persona = f"""Pretend that you are a {age}-year-old student. """

    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"Your gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "You are eligible for free school meals or pupil premium due to being financially disadvantaged. "
        
    return persona

def create_persona_3basic_thirdperson(x):
    age = int(x['age'])
    gender = int(x['Gender'])
    premium_pupil = x['PremiumPupil']

    if age == 8 or age == 11 or age == 18:
        persona = f"""Here's an {age}-year-old student. """
    else:
        persona = f"""Here's a {age}-year-old student. """

    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"The student's gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "The student is eligible for free school meals or pupil premium due to being financially disadvantaged. "
        
    return persona

def create_persona_duolingo_first_pov(x):
    persona = f"""Pretend that you are a person from {x['countries_full']}. """

    if x['client'] == "web":
        persona += f"You use a {x['client']} device. "
    else:
        persona += f"You use an {x['client']} device. "
        
    return persona

def create_persona_duolingo_third_pov(x):
    persona = f"""Here's a person from {x['countries_full']}. """

    if x['client'] == "web":
        persona += f"The person uses a {x['client']} device. "
    else:
        persona += f"The person uses an {x['client']} device. "
        
    return persona

def create_persona_wordbank_first_pov(x):
    age_months = int(x['age'])
    ethnicity = x['ethnicity'] if pd.notna(x['ethnicity']) else "unspecified"
    sex = x['sex'] if pd.notna(x['sex']) else "unspecified"
    mom_ed = x['mom_ed'] if pd.notna(x['mom_ed']) else "unspecified"
    article = "an" if str(age_months).startswith(('8', '11', '18')) else "a"
    
    persona = f"Pretend that you are {article} {age_months}-month-old child. "
    persona += f"Your sex is {sex} and your ethnicity is {ethnicity}. "
    persona += f"Your mother's education level is {mom_ed}. "

    return persona

def create_persona_3basic_plus_2relevant(x):
    age = int(x[0])
    gender = x[1]
    premium_pupil = x[2]
    math_anxiety = x[3]
    math_importance = x[4]
    if age == 8 or age == 11 or age == 18:
        persona = f"""Pretend that you are an {age}-year-old student. """
    else:
        persona = f"""Pretend that you are a {age}-year-old student. """
     
    gender = int(gender)
    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"Your gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "You are eligible for free school meals or pupil premium due to being financially disadvantaged. "
    if math_anxiety:
        persona += "You experience anxiety when facing math tasks. "
    else:
        persona += "You are confident when facing math tasks. "
    if math_importance:
        persona += "You are motivated and have a positive attitude towards math. "
    else:
        persona += "You lack motivation and have a negative attitude towards math. "
        
    return persona

def create_persona_3basic_plus_5relevant(x):
    age, gender, premium_pupil, numerical_proficiency, working_memory, math_anxiety, math_importance, parental_involvement = x
    
    if age in [8, 11, 18]:
        persona = f"Pretend that you are an {age}-year-old student. "
    else:
        persona = f"Pretend that you are a {age}-year-old student. "
    
    gender = int(gender)
    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"Your gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "You are eligible for free school meals or pupil premium due to being financially disadvantaged. "
    if numerical_proficiency:
        persona += "You have strong numerical skills. "
    else:
        persona += "You struggle with numerical tasks. "
    if working_memory:
        persona += "You have a strong working memory. "
    else:
        persona += "You find it challenging to hold information in your mind while working with it. "
    if math_anxiety:
        persona += "You experience anxiety when facing math tasks. "
    else:
        persona += "You are confident when facing math tasks. "
    if math_importance:
        persona += "You are motivated and have a positive attitude towards math. "
    else:
        persona += "You lack motivation and have a negative attitude towards math. "
    if parental_involvement:
        persona += "Your parents are highly involved in your learning. "
    else:
        persona += "Your parents are not very involved in your learning. "
    
    return persona

def create_persona_3basic_plus_5irrelevant(x):
    age, gender, premium_pupil, favorite_color, favorite_hobby, preferred_music, favorite_sport, preferred_reading = x
    
    if age in [8, 11, 18]:
        persona = f"Pretend that you are an {age}-year-old student. "
    else:
        persona = f"Pretend that you are a {age}-year-old student. "
    
    gender = int(gender)
    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"Your gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "You are eligible for free school meals or pupil premium due to being financially disadvantaged. "
    
    colors = ["blue", "green", "red", "yellow", "purple", "orange"]
    hobbies = ["reading", "cycling", "gaming", "drawing", "swimming"]
    music_types = ["pop", "rock", "classical", "jazz", "electronic", "hip hop"]
    sports = ["soccer", "basketball", "tennis", "swimming", "athletics"]
    reading_genres = ["fiction", "non-fiction", "fantasy", "mystery", "science fiction", "biography"]

    persona += f"Your favorite color is {colors[int(favorite_color)]}. "
    persona += f"Your favorite hobby is {hobbies[int(favorite_hobby)]}. "
    persona += f"You prefer listening to {music_types[int(preferred_music)]} music. "
    persona += f"Your favorite sport is {sports[int(favorite_sport)]}. "
    persona += f"You enjoy reading {reading_genres[int(preferred_reading)]} books. "
    
    return persona

def create_persona_3basic_plus_10(x):
    age, gender, premium_pupil, numerical_proficiency, working_memory, math_anxiety, math_importance, parental_involvement, favorite_color, favorite_hobby, preferred_music, favorite_sport, preferred_reading = x
    if age in [8, 11, 18]:
        persona = f"Pretend that you are an {age}-year-old student. "
    else:
        persona = f"Pretend that you are a {age}-year-old student. "
    
    gender = int(gender)
    gender_text = {0: "unspecified", 1: "female", 2: "male", 3: "other"}
    persona += f"Your gender is {gender_text[gender]}. "
    
    if premium_pupil:
        persona += "You are eligible for free school meals or pupil premium due to being financially disadvantaged. "

    if numerical_proficiency:
        persona += "You have strong numerical skills. "
    else:
        persona += "You struggle with numerical tasks. "
    if working_memory:
        persona += "You have a strong working memory. "
    else:
        persona += "You find it challenging to hold information in your mind while working with it. "
    if math_anxiety:
        persona += "You experience anxiety when facing math tasks. "
    else:
        persona += "You are confident when facing math tasks. "
    if math_importance:
        persona += "You are motivated and have a positive attitude towards math. "
    else:
        persona += "You lack motivation and have a negative attitude towards math. "
    if parental_involvement:
        persona += "Your parents are highly involved in your learning. "
    else:
        persona += "Your parents are not very involved in your learning. "

    colors = ["blue", "green", "red", "yellow", "purple", "orange"]
    hobbies = ["reading", "cycling", "gaming", "drawing", "swimming"]
    music_types = ["pop", "rock", "classical", "jazz", "electronic", "hip hop"]
    sports = ["soccer", "basketball", "tennis", "swimming", "athletics"]
    reading_genres = ["fiction", "non-fiction", "fantasy", "mystery", "science fiction", "biography"]

    persona += f"Your favorite color is {colors[int(favorite_color)]}. "
    persona += f"Your favorite hobby is {hobbies[int(favorite_hobby)]}. "
    persona += f"You prefer listening to {music_types[int(preferred_music)]} music. "
    persona += f"Your favorite sport is {sports[int(favorite_sport)]}. "
    persona += f"You enjoy reading {reading_genres[int(preferred_reading)]} books. "
    
    return persona