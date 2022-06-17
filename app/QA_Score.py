from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def QA_Score(summary, responses):
    true_ans = summary
    score = 0
    for r in responses:
        corpus = [true_ans, r]
        vect = TfidfVectorizer().fit_transform(corpus)
        score += ((cosine_similarity(vect[0], vect[1])))
    score /=len(responses)
    print(score)
    if(score>0.11):
        score = score*0.133 +0.598  
        
    score *= 100  
    score = int(score)
    print(score)
    return score