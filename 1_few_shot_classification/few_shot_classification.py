import torch
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from transformers import GPTQConfig
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    GPTQConfig,
    set_seed,
AwqConfig
)
import random


def instantiate_pipeline_qwen(
    access_token="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",
    model_id="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
):
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=access_token,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading GPTQ model... (this may take a while)")
    
    # Configure GPTQ settings
    gptq_config = GPTQConfig(
        bits=4,
        use_exllama=False,  # Disable exllama to avoid compatibility issues
        disable_exllama=True
    )
    
    # Load the model with explicit GPTQ configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 instead of auto
        trust_remote_code=True,
        token=access_token,
        quantization_config=gptq_config,
        low_cpu_mem_usage=True,  # This helps with memory management
        offload_buffers=True  # This helps avoid the meta device issue
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"✅ Pipeline instantiated for {model_id}!")
    return pipe


def classifier_qwen(text, pipe):
    '''
    Generates topic model labels from the provided topic model's keywords (topic_modeling_keywords) 
    with few-shot in-context learning using either llama or phi pre-trained model.
    
    Parameters:
    - text (str; required): The social media post to be classified.
    - pipe (object; required): The pipeline of an instantiated pre-trained model used.
    
    Returns:
    str: Labels of topic models with explanation and reasoning.
    '''
    system_message = '''You are a communication expert analyzing Facebook posts from various online community health networks. 
    These networks consist of organizations, agencies, or groups, and the posts may or may not include health-related content. 
    Your task is to classify each post based on its primary focus within the context of communication. 
    This classification will help identify the distinct communication strategies employed across these networks.
    '''

    task_prompt = '''Classify the given Facebook post from an organization, agency, or group within an online community health network 
    based on its primary focus in communication. Choose one of the following categories:

    1. Individual Behavior: Posts focused on behaviors or actions that affect a person's physical, mental, emotional, or social well-being. 
    This includes posts promoting individual actions, personal practices or tips, lifestyle changes, and messaging strategies aimed 
    at fostering behavior modifications for physical, mental, emotional, or social well-being..

    2. Policy or Environmental Approaches: Posts addressing policy initiatives or actions, such as laws, legislation, ordinances, 
    mandates, regulations, or rules. This also includes posts advocating for civic engagement, addressing the built environment, 
    or discussing economic or social surroundings.

    3. Other Organizational or Program/Service Information: Posts that share updates, resources, or services provided by the 
    organization or program, or that do not fall into "Individual Behavior" or "Policy or Environmental Approaches." This category 
    could include: 
        (1) promotion of the organization's services, events, or campaigns that do not include any behavioral component or action (No 1.), 
        (2) staff, volunteer, or partner spotlights, shout-outs, or appreciation posts, or 
        (3) success stories or stories highlighting the impact of a program or service.

    4. Irrelevant: Posts that do not fit any of the above categories. This includes season's greetings, holiday or observance recognition with no elements of No 1., No 2., or No 3.

    Strictly return the category number only (1, 2, 3, or 4). Below are examples to guide your classification.
    '''

    # Few-shot learning examples (unchanged)
    Example_1 = '''Did you know? 16% of teens have sustained Noise-Induced Hearing Loss due to prolonged exposure to personal listening devices. Adjust the noise level on your device to protect your hearing. On #WorldHearingDay, let's bring awareness to promoting ear & hearing care. Ear Peace Foundation'''
    output_1 = '1'
    
    Example_2 = '''Families, on Aug. 29, Miami-Dade County Mayor Daniella Levine Cava announced the HOMES Plan, which includes a full suite of programs that will provide relief to struggling homeowners and renters, create more housing people can afford by bringing new units online in the immediate short term and building new units, and preserve and enhance existing affordable housing. Find out more: '''
    output_2 = '2'
    
    Example_3 = '''According to the American Diabetes Association, 1.5 million people will be diagnosed with diabetes this year. If you have diabetes, let us help you manage and balance your health. We can provide support to all of your health care needs.'''
    output_3 = '3'

    Example_4='''-	During National Nutrition Month and beyond, choose delicious and nutrient-rich foods from each of the five basic food groups! If you want more information about your nutrition, schedule an appointment with one of our CHI Health registered dietitians: https://spr.ly/6182KlG9u.'''
    output_4='1'

    Example_5 = '''Trust President & CEO James R. Haj encourages support of our early child care educators this #TeacherAppreciationWeek. You can read his letter to editor in Miami's Community Newspapers here https://bit.ly/3KMbgdS #TeacherAppreciationWeek'''
    output_5 = "2"

    Example_6 = '''Today, we are spotlighting our friends at Girl Scouts Louisiana East who hosted their Believe in Girls (B.I.G.) event in April. The event was held at SE LA University and focused on STEM and hands-on experiences for 350 total Girl Scouts. This event was part of our Project-Based grants in 2021. We are grateful for partners like GSLE! #LiveUnited'''
    output_6 = '3'



    Example_7='''August is Summer Sun Safety Month! With the sun in full force, be mindful of its damaging effects and protect your skin from its harmful rays. Remember that sunscreen is your friend, so apply it often! #buildingbetterhabits #yourbestself #FLIPANY'''
    output_7='1'

    Example_8='''Governor Parsonâ€™s Supply Chain Task Force published its draft report today opening the public comment period. https://www.modot.org/supplychaintaskforce
    This report includes supply chain information, the task forceâ€™s findings, and recommended next steps for Missouri. The focus of the report is infrastructure needs and support for workforce to mitigate and minimize the impacts of supply chain challenges.
    Public comments will be received through June 17.'''
    output_8="2"



    Example_9='''We have a new blog, and it is a part of our Your Dollars at Work campaign! Read about our longtime partner Catholic Charities Diocese of Baton Rouge and how the ALICE Grant they received impacted their clients! Link: https://www.cauw.org/blog/your-dollars-work-impact-story-0 #LiveUnited'''
    output_9="3"

    Example_10='''-	Quick, easy, convenient and potentially life-saving. Schedule your free cancer screening today. View all upcoming screenings: https://bit.ly/3pEgReE Screening snapshot this week:
    Monroe - Colorectal, March 28, Ouachita Parish Health Unit
    Baton Rouge - Breast and Colorectal, March 30, Exxon Mobile YMCA
    Houma - Breast, Prostate and Colorectal, March 31, Best Buy'''
    output_10="1"



    Example_11='''Bi-State Regional Commission is hosting a transit summit and wants your input! Everyone is encouraged to complete this survey and/or attend a public meeting on Thursday 6/23 in Davenport from 4:00-6:00 p.m. Please complete the survey here, https://buff.ly/3xuaBsU'''
    output_11="2"



    Example_12='''-	CHRONIC DISEASE SELF-MANAGEMENT WORKSHOP hosted by: Independent Living Center of Southeast Missouri & University of Missouri Extension Registering now for the Spring 2023 class at the Doniphan location NO CHARGE!!!! Classes meet one time a week for six weeks â€¦..Approx. 2 Â½ hrs. Class size is limited.  â€¢ Living a Healthy Life is a FREE six-week workshop for adults living with chronic conditions and their family members, thanks to federal grant funding. â€¢ During this workshop you will learn how to:
     - Identify the latest pain management approaches
     - Manage fatigue and stress more effectively
     - Find solutions to problems caused by your condition
     - Identify ways to deal with anger, fear, frustration, anxiety and depression
     - Discuss the role of exercise and nutrition in chronic disease management â€¢ How to Communicate with family and friends
     - Form a partnership with your health-care team
     Please pre-register by calling Suzann McKnight at ILCSEMO   573-686-2333  Ext. 222 or John Fuller at Mo. University of Extension Center 573-686-8064.
     '''
    

    output_12='1'

    Example_13='''-	The agreement provides many admissions advantages and options for students to transfer up to 8 credits from their undergraduate coursework to satisfy specific course requirements for the MSEP program'''
    output_13='2'

    Example_14='''-	Scrambling to pick up all your last minute Christmas meal groceries? Come shop with us this morning behind Pennington Biomedical from 8am - 12pm! You won't find any shopping cart traffic jams or long check out lines, just #farmfresh fruits and veggies, #localfood, sweet treats and smiling faces!'''
    output_14='3'

    Example_15='''-	Tonight at 6 PM! Join us virtually to provide your ideas on improving mobility within your communities in Miami-Dade County.
    Learn about the Miami-Dade County New Mobility Initiative, a collaboration between Miami-Dade County Government, Urban Health Partnerships (UHP), Knight Foundation and Ford Motor Company's City:One program. The initiative aims to engage residents in the process of bringing new, innovative mobility solutions that can improve accessibility and equity in mobility for all County residents. 
    Register and learn more by visiting https://qrco.de/mdc1
    #MDCNewMobility #MiamiDade #MiamiDadeCounty #Mobility #Transit #Transportation #Technology #Commute #Community #PublicHealth #Accessibility #Equity #HealthyStreets #PublicSpaces #UrbanHealth #UHP #Ford'''
    
    output_15='2'

    Example_16='''-	The start of the school year has been amazing! And it has given us time to calculate the impact of our 2022 Summer.
    176 Youth engaged in FYB Programs
    Participating 2056:51 hours
    145 bikes were earned and 112 Badges were earned.
    15 interns had summer jobs through MYWE and FYB.
    4 trip to YMCA Swim classes
    16 bike rides
    And memories that last a lifetime!
    Thank you to all our students, parents, volunteers, and kind donors for making this happen. To learn how you can get involved go to frontyardbikes.com or email frontyardbikes@gmail.com
    Special thanks to: @mayorbroome @bigbuddyprogram @line4linebr @kanobike @knockknockchildrensmuseum @blackbirdletterpress @threeoclockproject @youthcitylab_br'''

    output_16='3'

    # Build the messages exactly as you designed
    prompt = f'''Now, classify the following post: {text}'''

    pairs = [
    ({"role":"user","content": Example_1},  {"role":"assistant","content": output_1}),
    ({"role":"user","content": Example_2},  {"role":"assistant","content": output_2}),
    ({"role":"user","content": Example_3},  {"role":"assistant","content": output_3}),
    ({"role":"user","content": Example_4},  {"role":"assistant","content": output_4}),
    ({"role":"user","content": Example_5},  {"role":"assistant","content": output_5}),
    ({"role":"user","content": Example_6},  {"role":"assistant","content": output_6}),
    ({"role":"user","content": Example_7},  {"role":"assistant","content": output_7}),
    ({"role":"user","content": Example_8},  {"role":"assistant","content": output_8}),
    ({"role":"user","content": Example_9},  {"role":"assistant","content": output_9}),
    ({"role":"user","content": Example_10}, {"role":"assistant","content": output_10}),
    ({"role":"user","content": Example_11}, {"role":"assistant","content": output_11}),
    ({"role":"user","content": Example_12}, {"role":"assistant","content": output_12}),
    ({"role":"user","content": Example_13}, {"role":"assistant","content": output_13}),
    ({"role":"user","content": Example_14}, {"role":"assistant","content": output_14}),
    ({"role":"user","content": Example_15}, {"role":"assistant","content": output_15}),
    ({"role":"user","content": Example_16}, {"role":"assistant","content": output_16}),
    ]
    random.seed(42)  
    random.shuffle(pairs)
    message = [
        {"role":"system","content": system_message},
        {"role":"user","content": task_prompt},
    ]
    for u,a in pairs:
        message.extend([u,a])
    message.append({"role": "user", "content": prompt})
    # Ensure pad token id is set (Qwen often uses eos as pad)
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    # Convert chat messages to a single string using Qwen's chat template
    chat_str = pipe.tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,  # model should start generating as assistant
    )

    # Model inference (greedy, 1 token)
    outputs = pipe(
        chat_str,
        max_new_tokens=100,
        do_sample=False,
        temperature=0,
        pad_token_id=pipe.tokenizer.pad_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id
    )
    #print(outputs)

    # Pipeline returns a list of dicts; 'generated_text' is a string
    gen = outputs[0]["generated_text"].strip()

    # Extract the first digit 1-4; fallback to "4" if model slips
    m = re.search(r"[1234]", gen)
    label = m.group(0) if m else "4"

#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

    return label

