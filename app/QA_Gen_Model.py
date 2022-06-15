from typing import Tuple
import nltk as nlp
import numpy as np


class QA_Gen_Model():
    
  
    
   

    def generate_test(summary) -> Tuple[list, list]:
        grammar = """
			CHUNK: {<NN>+<IN|DT>*<NN>+}
			{<NN>+<IN|DT>*<NNP>+}
			{<NNP>+<NNS>*}
		"""
        question_pattern = [
			"Explain in detail ",
			"Define ",
			"Write a short note on ",
			"What do you mean by "
		]
        sentences = nlp.sent_tokenize(summary)
        cp = nlp.RegexpParser(grammar)
        question_answer_dict = dict()
        
        
            
        for sentence in sentences:
            tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
            tree = cp.parse(tagged_words)
            temp =''
            for subtree in tree.subtrees():
                temp = ""
                if subtree.label() == "CHUNK":
                    for sub in subtree:
                        temp += sub[0]
                        temp += " "
                    temp = temp.strip()
                    temp = temp.upper()
                    if temp not in question_answer_dict:
                        # if len(nlp.word_tokenize(sentence)) > 20:
                        #     question_answer_dict[temp] = sentence
                        # else:
                        question_answer_dict[temp] = sentence
                            
        keyword_list = list(question_answer_dict.keys())
        question_answer = list()
        
        for _ in range(5):
            rand_num = np.random.randint(0, len(keyword_list))
            selected_key = keyword_list[rand_num]
            answer = question_answer_dict[selected_key]
            rand_num %= 4
            question = question_pattern[rand_num] + selected_key + "."
            question_answer.append({"Question": question, "Answer": answer})
        
        que = list()
        ans = list()
        while len(que) < 5 :
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in que:
                que.append(question_answer[rand_num]["Question"])
                ans.append(question_answer[rand_num]["Answer"])
            else:
                continue
        return que, ans	
        
        

    
    
    	
        
  
		


