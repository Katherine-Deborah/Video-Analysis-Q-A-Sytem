import torch

def answer_question(question, context, models):
    """Answer question based on context using a Gemma-style LLM"""

    try:
        prompt = f"""Based on the following video analysis data, please answer the question.

Video Captions and Transcription:
{context}

Question: {question}

Please provide a clear and concise answer based only on the information provided above."""

        messages = [{"role": "user", "content": prompt}]

        text = models['qa_tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = models['qa_tokenizer'](
            [text],
            return_tensors="pt"
        ).to(models['qa_model'].device)

        with torch.no_grad():
            generated_ids = models['qa_model'].generate(
                **model_inputs,
                max_new_tokens=32768,
                # do_sample=False,
                # temperature=0.7,
                # eos_token_id=models['qa_tokenizer'].eos_token_id
            )

        # Extract the new tokens after the input prompt
        input_length = model_inputs.input_ids.shape[-1]
        output_ids = generated_ids[0][input_length:].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        answer = models['qa_tokenizer'].decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return answer

    except Exception as e:
        return f"Error generating answer: {e}"

    
def get_context_for_qa(session_id, conn):
    """Retrieve all captions and transcriptions for QA context"""
    cursor = conn.cursor()
    
    # Get captions
    cursor.execute(
        "SELECT timestamp, caption FROM captions WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    captions = cursor.fetchall()
    
    # Get transcription
    cursor.execute(
        "SELECT transcription FROM transcriptions WHERE session_id = ?",
        (session_id,)
    )
    transcription_result = cursor.fetchone()
    
    context = "CAPTIONS:\n"
    for timestamp, caption in captions:
        context += f"At {timestamp:.1f}s: {caption}\n"
    
    if transcription_result:
        context += f"\nAUDIO TRANSCRIPTION:\n{transcription_result[0]}"
    
    # print(context)
    
    return context