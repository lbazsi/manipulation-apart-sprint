# AI Manipulation Apart Sprint

## 🚀 *DEEB-learn*: Benchmark to quantify tonal shift from LLM response based on contextual changes

## 🌟 Overview  

- Large language models can shift tone under minor contextual framing changes  
- This complicates the evaluation of model outputs 
- We present a benchmark to quantify this shift  
- Our approach combines mechanistic interpretability insights, AI judge assembly and a classification benchmark  

## 🎯 Key Features  

- Uses multi-judge assembly to score with behavioral traits 
- Uses 2 context modifyers to study evalutation or oversight 
- Uses additional mechanistic interpretability information

## 🚀 Getting Started  
To get started using this project, use the deeb_dataset500.jsonl file and the thre_judges_trial1.py file. Make sure to run the files using the requirements.txt. This process will generate behavioral traits from three judges. Further analysis and exploration can be done using this data.

### Prerequisites  
- all prequisites are stored in the requirements.txt file

### Setup  
```bash
# Setup virtual environment
python3 -m venv test_env
source test_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### Run  
python three_judges_trial1.py
```

## Team  

- **Rick** 
- **Balazs**
- **Ama**
