import json
import requests
import re
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path

@dataclass
class EvaluationResult:
    mistake_identification: Dict[str, float]
    mistake_location: Dict[str, float] 
    providing_guidance: Dict[str, float]
    actionability: Dict[str, float]

class TutorResponseEvaluator:
    def __init__(self, api_base_url: str, api_key: str = None):
        """
        Initialize the evaluator
        
        Args:
            api_base_url: Base URL for the Qwen API (e.g., "http://localhost:8000/v1")
            api_key: API key if required
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.few_shot_examples = self._create_few_shot_examples()
        
    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples for prompt"""
        examples = """
**Training Examples for Expert Evaluation:**

**Example 1: Mathematical Error Correction**
Conversation Context: "Student: yes \n Tutor: So, we wan to start with the first 2 digits. \n Tutor: What is the value of 5 * 14?\n Student: 75"
Tutor Response: "I think you meant to say 70, since 5 times 14 is 70, not 75 - let's take a look at how we got that together."

Expert Analysis:
- Mistake Identification: Yes (tutor immediately recognizes the calculation error and states the correct answer)
- Mistake Location: Yes (specifically identifies the problematic calculation "5 * 14")
- Providing Guidance: No (offers to work together but doesn't provide concrete steps or explanation)
- Actionability: No (suggestion to "take a look" is vague without clear next steps)

**Example 2: Complex Problem Solving with Double-Counting Error**
Conversation Context: "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: Kamil wants to renovate his kitchen at home. For this purpose, he hired two professionals who work for him 6 hours a day for 7 days. What does it cost Kamil to hire these professionals if one of them is paid $15 per hour of work? \n Student: The two professionals work together for 6 x 2 = 12 hours a day.\nThey work for a total of 12 x 7 = 84 hours in 7 days.\nSo, the first professional earns a total of 15 x 84 = $1260 for the entire work.\nSince there are two professionals, Kamil pays a total of 1260 x 2 = $2520 for their entire work.\n 2520 \n Tutor: could you please walk me through your solution? \n Student: Sure. I started by calculating the total number of hours worked by both professionals. I multiplied 6 hours by 2 which gives us 12 hours a day. Then I multiplied 12 hours by 7 days which gives us 84 hours in total. Then I calculated the amount earned by the first professional for the entire work which is 15 x 84 = $1260. Finally since there are two professionals I multiplied 1260 by 2 which gives us a total of $2520 for their entire work. \n Tutor: But you have already accounted for both of them in the first step that is 12 hours per day. Why are you multiplying again in the last step? \n Student: Oh I see. I was multiplying the total amount earned by the first professional for the entire work which is 15 x 84 = $1260 by 2 to get the total amount paid by Kamil for both professionals. So the total amount paid by Kamil is $1260 x 2 = $2520."
Tutor Response: "You're on the right track, but remember that you already calculated the total hours worked by both professionals in the first step, so you don't need to multiply the first professional's earnings by 2."

Expert Analysis:
- Mistake Identification: Yes (clearly identifies the double-counting logical error)
- Mistake Location: Yes (precisely points to the redundant multiplication step)
- Providing Guidance: Yes (explains the logical connection between steps and the error)
- Actionability: Yes (provides clear instruction: "don't multiply by 2")

**Example 3: Ambiguous Response to Correct Solution**
Conversation Context: "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: In one of the building blocks at Oakland High there are 5 classes. Each class uses 2 whiteboards each and each whiteboard needs about 20ml of ink for a day's use. If ink costs 50 cents per ml, how much (in dollars) would it cost to use the boards for one day? \n Student: There are 5 classes and each class uses 2 whiteboards, so there are 5 x 2 = 10 whiteboards in total.\nEach whiteboard needs 20ml of ink, so 10 whiteboards need 10 x 20 = 200ml of ink in total.\nInk costs 50 cents per ml, so 200ml of ink would cost 200 x 0.5 = 100 cents.\nSince there are 100 cents in a dollar, it would cost $1 to use the boards for one day.\n 1 \n Tutor: so there are 5 x 2 = 10 whiteboards in total. Each whiteboard needs 20ml of ink so 10 whiteboards need 10 x 20 = 200ml of ink in total. Ink costs 50 cents per ml so 200ml of ink would cost 200 x 0.5 = 100 cents. Since there are 100 cents in a dollar it would cost $1 to use the boards for one day."
Tutor Response: "You're almost there! Remember, 100 cents is equal to $1, so the total cost would be $1."

Expert Analysis:
- Mistake Identification: To some extent (implies there's an issue with "almost there" but student solution appears correct)
- Mistake Location: No (doesn't identify any specific error location since solution is actually correct)
- Providing Guidance: To some extent (restates conversion fact but doesn't address actual issues)
- Actionability: No (doesn't provide clear next steps since the student's answer is correct)
"""
        return examples

    def _create_evaluation_prompt(self, conversation_history: str, tutor_response: str) -> str:
        """Create the evaluation prompt for the model"""
        
        prompt = f"""You are an experienced educational assessment specialist with expertise in analyzing tutoring interactions. Your role is to objectively evaluate the quality of tutor responses based on established pedagogical criteria.

As an expert evaluator, you must:
- Carefully analyze each tutor response for its educational effectiveness
- Apply consistent evaluation criteria across all examples
- Provide precise probability scores that reflect your confidence in each assessment
- Focus on the pedagogical value and student learning outcomes

Your task is to evaluate tutor responses across four critical dimensions:

1. **Mistake Identification**: Does the tutor recognize mistakes in the student's response?
   - Yes: mistake is clearly identified/recognized
   - To some extent: tutor suggests there may be a mistake but sounds uncertain
   - No: tutor does not recognize the mistake

2. **Mistake Location**: Does the tutor accurately point to the mistake location?
   - Yes: clearly points to exact location of genuine mistake
   - To some extent: shows some awareness but is vague/unclear
   - No: doesn't provide details about mistake location

3. **Providing Guidance**: Does the tutor offer correct and relevant guidance?
   - Yes: provides correct and relevant guidance
   - To some extent: guidance is partially incorrect/incomplete/misleading
   - No: no guidance provided or guidance is irrelevant/incorrect

4. **Actionability**: Is the feedback actionable with clear next steps?
   - Yes: provides clear suggestions for what student should do next
   - To some extent: indicates something needs to be done but unclear what
   - No: doesn't suggest any action

{self.few_shot_examples}

Now, as an expert educational assessment specialist, carefully evaluate this new tutoring interaction:

**Conversation Context:** {conversation_history}

**Tutor Response to Evaluate:** {tutor_response}

**Instructions for Evaluation:**
1. Read the conversation carefully to understand the student's mistake
2. Analyze the tutor's response against each of the four dimensions
3. Consider the pedagogical effectiveness and learning impact
4. Assign probability scores based on your expert judgment

Provide your expert evaluation in the following JSON format with confidence scores (0.0-1.0) for each category:

{{
    "mistake_identification": {{
        "Yes": 0.0,
        "To some extent": 0.0,
        "No": 0.0
    }},
    "mistake_location": {{
        "Yes": 0.0,
        "To some extent": 0.0,
        "No": 0.0
    }},
    "providing_guidance": {{
        "Yes": 0.0,
        "To some extent": 0.0,
        "No": 0.0
    }},
    "actionability": {{
        "Yes": 0.0,
        "To some extent": 0.0,
        "No": 0.0
    }}
}}

**Critical Evaluation Requirements:**
- All confidence scores for each dimension must sum to 1.0
- Base your assessment on pedagogical best practices and educational research
- Consider the learning impact on the student
- Maintain objectivity and consistency with the provided examples
- Your evaluation should reflect expert-level analysis of tutoring effectiveness"""

        return prompt

    def _call_qwen_api(self, prompt: str) -> str:
        """
        Call Qwen3-235B API
        
        Args:
            prompt: The input prompt
            
        Returns:
            Model response as string
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": "Qwen3-235B-A22B",  # Adjust model name as needed
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            return None

    def _parse_response(self, response: str) -> EvaluationResult:
        """
        Parse the model response and extract probabilities
        
        Args:
            response: Raw model response
            
        Returns:
            EvaluationResult object with probabilities
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            result_dict = json.loads(json_str)
            
            # Normalize probabilities to ensure they sum to 1.0
            for dimension in result_dict:
                total = sum(result_dict[dimension].values())
                if total > 0:
                    for label in result_dict[dimension]:
                        result_dict[dimension][label] /= total
                else:
                    # If all zeros, set equal probabilities
                    for label in result_dict[dimension]:
                        result_dict[dimension][label] = 1.0/3
            
            return EvaluationResult(
                mistake_identification=result_dict["mistake_identification"],
                mistake_location=result_dict["mistake_location"],
                providing_guidance=result_dict["providing_guidance"],
                actionability=result_dict["actionability"]
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            
            # Return default equal probabilities
            default_probs = {"Yes": 1/3, "To some extent": 1/3, "No": 1/3}
            return EvaluationResult(
                mistake_identification=default_probs.copy(),
                mistake_location=default_probs.copy(),
                providing_guidance=default_probs.copy(),
                actionability=default_probs.copy()
            )

    def evaluate_sample(self, sample: Dict) -> EvaluationResult:
        """
        Evaluate a single sample
        
        Args:
            sample: Dictionary containing conversation_history and response
            
        Returns:
            EvaluationResult with probabilities
        """
        prompt = self._create_evaluation_prompt(
            sample["conversation_history"],
            sample["response"]
        )
        
        response = self._call_qwen_api(prompt)
        if response is None:
            # Return default probabilities if API call fails
            default_probs = {"Yes": 1/3, "To some extent": 1/3, "No": 1/3}
            return EvaluationResult(
                mistake_identification=default_probs.copy(),
                mistake_location=default_probs.copy(),
                providing_guidance=default_probs.copy(),
                actionability=default_probs.copy()
            )
        
        return self._parse_response(response)

    def _load_existing_results(self, output_file: str) -> Tuple[List[Dict], int]:
        """
        Load existing results from file and return the data and last processed index
        
        Args:
            output_file: Path to the output file
            
        Returns:
            Tuple of (existing_results_list, last_processed_index)
        """
        if not os.path.exists(output_file):
            return [], -1
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if not existing_data:
                return [], -1
            
            # Find the highest sample_index
            last_index = max(item.get("sample_index", -1) for item in existing_data)
            print(f"Found existing results up to index {last_index}")
            return existing_data, last_index
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading existing results: {e}")
            return [], -1

    def _save_single_result(self, result: EvaluationResult, sample: Dict, sample_index: int, output_file: str):
        """
        Save a single result to the output file (append or update)
        
        Args:
            result: EvaluationResult object
            sample: Original sample data
            sample_index: Index of the sample
            output_file: Output file path
        """
        # Load existing results
        existing_results, _ = self._load_existing_results(output_file)
        
        # Create new result entry
        new_entry = {
            "sample_index": sample_index,
            "original_response": sample.get("response", ""),
            "conversation_history": sample.get("conversation_history", ""),
            "evaluation_results": {
                "mistake_identification": result.mistake_identification,
                "mistake_location": result.mistake_location,
                "providing_guidance": result.providing_guidance,
                "actionability": result.actionability
            }
        }
        
        # Check if this sample_index already exists
        updated = False
        for i, existing_entry in enumerate(existing_results):
            if existing_entry.get("sample_index") == sample_index:
                existing_results[i] = new_entry
                updated = True
                break
        
        # If not updated, append new entry
        if not updated:
            existing_results.append(new_entry)
        
        # Sort by sample_index to maintain order
        existing_results.sort(key=lambda x: x.get("sample_index", 0))
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

    def evaluate_dataset(self, dataset: List[Dict], output_file: str, delay: float = 1.0, start_from: int = None) -> List[EvaluationResult]:
        """
        Evaluate entire dataset with incremental saving
        
        Args:
            dataset: List of samples to evaluate
            output_file: Output file path for saving results
            delay: Delay between API calls to avoid rate limiting
            start_from: Index to start from (if None, will auto-detect from existing results)
            
        Returns:
            List of EvaluationResult objects
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load existing results to determine where to start
        existing_results, last_processed_index = self._load_existing_results(output_file)
        
        if start_from is not None:
            start_index = start_from
        else:
            start_index = last_processed_index + 1
        
        print(f"Starting evaluation from index {start_index}")
        print(f"Total samples to process: {len(dataset)}")
        print(f"Remaining samples: {max(0, len(dataset) - start_index)}")
        
        results = []
        
        for i in range(len(dataset)):
            if i < start_index:
                # Skip already processed samples, but add placeholder results
                default_probs = {"Yes": 1/3, "To some extent": 1/3, "No": 1/3}
                results.append(EvaluationResult(
                    mistake_identification=default_probs.copy(),
                    mistake_location=default_probs.copy(),
                    providing_guidance=default_probs.copy(),
                    actionability=default_probs.copy()
                ))
                continue
            
            print(f"Evaluating sample {i+1}/{len(dataset)} (Index: {i})")
            
            try:
                # Evaluate the sample
                result = self.evaluate_sample(dataset[i])
                results.append(result)
                
                # Save result immediately
                self._save_single_result(result, dataset[i], i, output_file)
                print(f"✓ Sample {i} evaluated and saved")
                
                # Add delay to avoid rate limiting
                if i < len(dataset) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"✗ Error processing sample {i}: {e}")
                # Add default result for failed samples
                default_probs = {"Yes": 1/3, "To some extent": 1/3, "No": 1/3}
                result = EvaluationResult(
                    mistake_identification=default_probs.copy(),
                    mistake_location=default_probs.copy(),
                    providing_guidance=default_probs.copy(),
                    actionability=default_probs.copy()
                )
                results.append(result)
                
                # Still save the failed result
                self._save_single_result(result, dataset[i], i, output_file)
                
                # Continue with next sample
                continue
        
        return results

    def save_results(self, results: List[EvaluationResult], dataset: List[Dict], output_file: str):
        """
        Save results to JSON file, including original responses for identification
        (This method is kept for compatibility but evaluate_dataset now saves incrementally)
        
        Args:
            results: List of evaluation results
            dataset: Original dataset containing the responses
            output_file: Output file path
        """
        output_data = []
        
        for i, result in enumerate(results):
            # Get the original response if available
            original_response = dataset[i].get("response", "") if i < len(dataset) else ""
            
            output_data.append({
                'sent_id': dataset[i].get("sent_id", "") if i < len(dataset) else "",
                "sample_index": i,
                "original_response": original_response,
                "conversation_history": dataset[i].get("conversation_history", "") if i < len(dataset) else "",
                "evaluation_results": {
                    "mistake_identification": result.mistake_identification,
                    "mistake_location": result.mistake_location,
                    "providing_guidance": result.providing_guidance,
                    "actionability": result.actionability
                }
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    # Initialize evaluator
    evaluator = TutorResponseEvaluator(
        api_base_url="http://qwen3-235b-a22b.bd-ai-llm.mlops-infer.tal.com/v1",
        api_key="EMPTY"
    )

    # Load data
    data_file = '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/data_new/train_qwen3.json'
    output_file = "/mnt/cfs/huangzhiwei/Data_mining_BAE2025/projects_update_0610/qwen-create-soft-labels/evaluation_results.json"
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {data_file}")
    
    # Evaluate dataset with incremental saving
    # This will automatically detect where to resume from if the process was interrupted
    results = evaluator.evaluate_dataset(data, output_file, delay=1.0)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to {output_file}")
    print(f"Total samples processed: {len(results)}")

if __name__ == "__main__":
    main()