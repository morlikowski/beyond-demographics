from typing import Dict


TEXTUAL_OFFENSIVENESS = {
            "prompt": "Is the following text benign or offensive from the perspective of someone who is {attributes}?\n\n"
            "Text: {text}\n### Answer: ",
            "text_only": "Is the following text benign or offensive?\n\n Text: {text}\n### Answer: ",
            "response_split": "### Answer: ",
        }

GENERIC = {
            "prompt": "Annotator: {attributes}\nText: {text}\n### Label: ",
            "text_only": "Text: {text}\n### Label: ",
            "response_split": "### Label: ",
        }

FOR_SEQUENCE_CLASSIFICATION = {
            "prompt": "Annotator: {attributes}\nText: {text}",
            "text_only": "{text}"
        }

INTIMACY_5_POINT = {
    "prompt":   '**Your Profile**: {attributes}\n'
                '**Question**: Consider you read this post on Twitter, how intimate do you think it is?\n'
                '**Post**: {text}\n'
                '(A) not intimate at all\n(B) barely intimate\n(C) somehat intimate\n(D) moderately intimate\n(E) very intimate\n'
                '**Answer**: (',
    'text_only':    '**Question**: Consider you read this post on Twitter, how intimate do you think it is?\n'
                    '**Post**: {text}\n'
                    '(A) not intimate at all\n(B) barely intimate\n(C) somehat intimate\n(D) moderately intimate\n(E) very intimate\n'
                    '**Answer**: (',
    # chat prompt includes different parts that are composed depending on configuration (content only, attributes) in Prompter.generate_chat_prompt
    'chat': {
                'system': '{attributes}\n',
                'user':
                    '**Question**: Consider you read this post on Twitter, how intimate do you think it is?\n'
                    '**Post**: {text}\n'
                    '(A) not intimate at all\n(B) barely intimate\n(C) somehat intimate\n(D) moderately intimate\n(E) very intimate\n',
                'assistant': '**Answer**: ('
        },
    "response_split": "**Answer**: ",
}

POLITENESS_5_POINT = {
     'prompt':  '**Your Profile**: {attributes}\n'
                '**Question**: Consider you read this email from a colleague, how polite do you think it is?\n'
                '**Email:**: {text}\n'
                '(A) not polite at all\n(B) barely polite\n(C) somewhat polite\n(D) moderately polite\n(E) very polite\n'
                '**Answer**: (',
     'text_only':   '**Question**: Consider you read this email from a colleague, how polite do you think it is?\n'
                    '**Email:**: {text}\n'
                    '(A) not polite at all\n(B) barely polite\n(C) somewhat polite\n(D) moderately polite\n(E) very polite\n'
                    '**Answer**: (',
    # chat prompt includes different parts that are composed depending on configuration (content only, attributes) in Prompter.generate_chat_prompt
    'chat': {
                'system': '{attributes}\n',
                'user':
                    '**Question**: Consider you read this email from a colleague, how polite do you think it is?\n'
                    '**Email:**: {text}\n'
                    '(A) not polite at all\n(B) barely polite\n(C) somewhat polite\n(D) moderately polite\n(E) very polite\n',
                'assistant': '**Answer**: ('
            },
    'response_split': "**Answer**: "
}

SAFETY_3_POINT = {
     'prompt':  '**Your Profile**: {attributes}\n'
                '**Question**: Consider you read this conversation between a chat bot and a human, do you think it is safe or unsafe in terms of harmful content, unfair bias or safety policy violations?\n'
                '**Conversation:**: {text}\n'
                '(A) safe\n(B) unsure\n(C) unsafe\n'
                '**Answer**: (',
       'text_only': '**Question**: Consider you read this conversation between a chat bot and a human, do you think it is safe or unsafe in terms of harmful content, unfair bias or safety policy violations?\n'
                    '**Conversation:**: {text}\n'
                    '(A) safe\n(B) unsure\n(C) unsafe\n'
                    '**Answer**: (',
        # chat prompt includes different parts that are composed depending on configuration (content only, attributes) in Prompter.generate_chat_prompt
        'chat': {
                'system': '{attributes}\n',
                'user':
                    '**Question**: Consider you read this conversation between a chat bot and a human, do you think it is safe or unsafe in terms of harmful content, unfair bias or safety policy violations?\n'
                    '**Conversation:**: {text}\n'
                    '(A) safe\n(B) unsure\n(C) unsafe\n',
                'assistant': '**Answer**: ('
            },
    'response_split': "**Answer**: "
}

OFFENSIVENESS_5_POINT = {
     'prompt':  '**Your Profile**: {attributes}\n'
                '**Question**: Consider you read this comment on Reddit, how offensive do you think it is?\n'
                '**Comment:**: {text}\n'
                '(A) not offensive at all\n(B) barely offensive\n(C) somewhat offensive\n(D) moderately offensive\n(E) very offensive\n'
                '**Answer**: (',
     'text_only':   '**Question**: Consider you read this comment on Reddit, how offensive do you think it is?\n'
                    '**Comment:**: {text}\n'
                    '(A) not offensive at all\n(B) barely offensive\n(C) somewhat offensive\n(D) moderately offensive\n(E) very offensive\n'
                    '**Answer**: (',
    # chat prompt includes different parts that are composed depending on configuration (content only, attributes) in Prompter.generate_chat_prompt
    'chat': {
                'system': '{attributes}\n',
                'user':
                    '**Question**: Consider you read this comment on Reddit, how offensive do you think it is?\n'
                    '**Comment:**: {text}\n'
                    '(A) not offensive at all\n(B) barely offensive\n(C) somewhat offensive\n(D) moderately offensive\n(E) very offensive\n',
                'assistant': '**Answer**: ('
            },
    'response_split': "**Answer**: "
}

SENTIMENT_5_POINT = {
    "prompt":   '**Your Profile**: {attributes}\n'
                '**Question**: Consider you read this text, what do you think is the sentiment it expresses?\n'
                '**Text**: {text}\n'
                '(A) very negative\n(B) somewhat negative\n(C) neutral\n(D) somewhat positive\n(E) very positive\n'
                '**Answer**: (',
    'text_only':    '**Question**: Consider you read this text, what do you think is the sentiment it expresses?\n'
                    '**Text**: {text}\n'
                    '(A) very negative\n(B) somewhat negative\n(C) neutral\n(D) somewhat positive\n(E) very positive\n'
                    '**Answer**: (',
    # chat prompt includes different parts that are composed depending on configuration (content only, attributes) in Prompter.generate_chat_prompt
    'chat': {
                'system': '{attributes}\n',
                'user':
                    '**Question**: Consider you read this text, what do you think is the sentiment it expresses?\n'
                    '**Text**: {text}\n'
                    '(A) very negative\n(B) somewhat negative\n(C) neutral\n(D) somewhat positive\n(E) very positive\n',
                'assistant': '**Answer**: ('
            },
    "response_split": "**Answer**: "
}

TASK_MAP = {
    'diaz': SENTIMENT_5_POINT,
    'intimacy': INTIMACY_5_POINT,
    'offensiveness': OFFENSIVENESS_5_POINT,
    'politeness': POLITENESS_5_POINT,
    'dices-990': SAFETY_3_POINT,
    'dices-350': SAFETY_3_POINT
}

class Prompter(object):

    def __init__(self, attribute_token_maps, task=None, is_chat=False, attribute_format=None):
        if is_chat and not task:
            raise ValueError('Can not use chat templating when formatting for sequence classification (prediction head)')
        if task:
            self.template = TASK_MAP[task]
        else:
            self.template = FOR_SEQUENCE_CLASSIFICATION
        self.attribute_token_maps = attribute_token_maps
        self.is_chat = is_chat
        self.attribute_format = attribute_format

    def _format_attributes(
            self,
            attributes: Dict[str, str]
    ):
        
        if 'user_id' in attributes:
            if self.is_chat:
                identifier = f'You have the unique identifier {attributes["user_id"]}'
            else:
                identifier = f'unique identifier {attributes["user_id"]}'
            if len(attributes) == 1:
                return identifier
        
        race = attributes['race']
        if race == 'pacific islander':
            race = f'a {race}'
        gender = attributes['gender']
        if gender == 'man' or gender == 'woman':
            gender = f'a {gender}'
        age = attributes['age']
        edu = attributes['education']
        if edu == 'college degree' or edu == 'graduate degree':
            edu = f'a {edu}'
        
        if self.attribute_format:
            if self.attribute_format == 'long':
                attribute_str = (
                    f'In terms of race or ethnicity, you are {race}.' 
                    f' You are {age}.' 
                    f' In terms of gender, you are {gender}.'
                    f' The highest degree or level of school that you have completed is {edu}.'
                )
                if 'user_id' in attributes:
                    attribute_str = f'{identifier}. {attribute_str}'
                return attribute_str
            else:
                raise NotImplementedError()
        else:
            attribute_str = f'{race}, {age}, {gender}, {edu}'
            if self.is_chat:
                attribute_str = f'You are {attribute_str}'
                if 'user_id' in attributes:
                    attribute_str = f'{identifier}. {attribute_str}'
            else:
                if 'user_id' in attributes:
                    attribute_str = f'{identifier}, {attribute_str}'
            return attribute_str
    
    def generate_prompt(
        self,
        text: str,
        attributes: Dict[str, str] = {},
        task=None
    ) -> str:
        if task:
            template = TASK_MAP[task]
        else:
            template = self.template

        if not attributes:
            return template["text_only"].format(
            text=text
        )
        attribute_str = self._format_attributes(attributes)
        prompt = template["prompt"].format(
            text=text,
            attributes=attribute_str
        )
        return prompt
    
    def generate_chat_prompt(
        self,
        text: str,
        attributes: Dict[str, str] = {},
        task=None
    ) -> str:
        if task:
            template = TASK_MAP[task]
        else:
            template = self.template
        prompt = [{
                    'role': 'user',
                    'content': template['chat']['user'].format(text=text)
                }]
        
        if attributes:
            attribute_str = self._format_attributes(attributes)
            prompt = [
                {
                    'role': 'system',
                    'content': template['chat']['system'].format(attributes=attribute_str)
                }
            ] + prompt

        prompt = prompt + [
                {
                    'role': 'assistant',
                    'content': template['chat']['assistant']
                }
            ]

        return prompt
    
    def apply_prompt(self, data_point, ignore_task=False):
        attribute_tokens = {}
        for attribute_name, token_map in self.attribute_token_maps.items():
            value = data_point[attribute_name]
            tokens = token_map[value]
            tokens_str = ' '.join(tokens)
            attribute_tokens[attribute_name] = tokens_str
        if self.is_chat:
            full_prompt = self.generate_chat_prompt(
                data_point['text'],
                attributes=attribute_tokens,
                task=None if ignore_task else data_point['task']
            )
        else:
            full_prompt = self.generate_prompt(
                data_point['text'],
                attributes=attribute_tokens,
                task=None if ignore_task else data_point['task']
            )

        return full_prompt
    

class PrompterForGeneration(Prompter):

    def __init__(self, attribute_token_maps):
        super().__init__(attribute_token_maps)
        self.template = GENERIC

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()