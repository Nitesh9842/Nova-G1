from googlesearch import search

def GoogleSearch(prompt):
    results = list(search(prompt, advanced=True, num_results=5))
    Answer = f"the search results for '{prompt}' are :\n [start]\n"

    for i in results :
        Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"   # type: ignore

    Answer +="[end]"
    # print(Answer)      
    return Answer


def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [lines for lines in lines if lines.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer 
 

if __name__ == "__main__":
    while True:
        prompt = input("Enter your Query :  ")
        print(GoogleSearch(prompt))  
