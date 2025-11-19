import ollama
import torch
import time

def run_llama(des):
    department_content = 'Please remove all the information related to clothing color, texture, logo and pattern from the input sentences, such as tonal stitching.'\
    'Only retain the information that describes the clothing shape.'\
    'Organize the remaining information into a complete sentence. The sentence begins with type of the clothing.'\
    'The format of this sentence is "[cloth category] with [attribute], [attribute], [attribute], ......". '\
    '[cloth category] must be a word that appears in the input sentence, indicating a type of clothing. [Attribute] is an attribute that describes the clothing shape.'\
    'For example, if the input is Short sleeve cotton piqu&eacute polo in heathered grey. Ribbed spread collar and sleeve cuffs. Two-button placket. Multicolored zebra logo patch at breast. Slits at side seams. Tonal stitching.'\
    'The output should be Polo with short sleeve, ribbed spread collar, sleeve cuffs, slits at side seams.'\
    'If the input is Sleeveless relaxed-fit t-shirt in black. Sheer mesh tonal exterior layer throughout. Crewneck collar. Button-loop closure at back. 3D-style floral print throughout in tones of grey, blue, and red. Curved hem. Tonal stitching.'\
    'The output should be T-shirt with sleeveless, crewneck collar, relaxed-fit, curved hem.'\
    'Just output the sentence without starting with "Here is" or including any negation words like "no information".'\
    'There should be no hints, like "Here is the rewritten sentence:".'

    start = time.time()
    response_de = ollama.chat(model='llama3:8b', messages=[
        {"role": "system", "content": department_content},
        {
            'role': 'user',
            'content': 'input sentence: ' + des},
    ])
    department = response_de['message']['content']
    end = time.time()
    return department

def get_mode(des):
    department_content = '''
    Please determine whether the input text belongs to the color, shape, or comb mode.
    color: The input text describes only the garment's color, texture, or material attributes, such as "t-shirt in red" or "suede shirt."
    shape: The input text describes only the garment's shape attributes, such as "shirt with long sleeves" or "t-shirt with a V-neck."
    comb: The input text describes both the garment's shape attributes and its color, texture, or material attributes, such as "long-sleeve red shirt with a V-neck.", "denim t-shirt with round collar."
    Only output the mode, without explanations or any additional text.
    '''

    response_de = ollama.chat(model='llama3:8b', messages=[
        {"role": "system", "content": department_content},
        {
            'role': 'user',
            'content': 'input text: ' + des,
            },
    ], 
    options={
        "temperature": 0
        })
    department = response_de['message']['content']
    return department

def get_seg_prompt(des):
    department_content = '''
    Please judge which category the subject in the input text belongs to from the following list: 
    [SHIRTS, TOPS, SWEATERS, JEANS, PANTS, SHORTS, SKIRTS, DRESSES, HATS, HAIRS, SHOES]. 
    Output only the category, without any explanations or additional text.
    '''

    response_de = ollama.chat(model='llama3:8b', messages=[
        {"role": "system", "content": department_content},
        {
            'role': 'user',
            'content': 'input text: ' + des,
            },
    ], 
    options={
        "temperature": 0
        })
    department = response_de['message']['content']
    return department