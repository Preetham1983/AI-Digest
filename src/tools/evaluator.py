from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.models.items import IngestedItem, EvaluationResult
from src.services.llm import llm
from src.services.logger import logger

class BaseEvaluator(ABC):
    @abstractmethod
    def get_persona_name(self) -> str:
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        pass

    async def evaluate_batch(self, items: List[IngestedItem]) -> List[EvaluationResult]:
        persona = self.get_persona_name()
        if not items:
            return []
            
        # Format batch prompt
        items_str = "\n\n".join([item.to_prompt_string() for item in items])
        prompt = self.get_batch_prompt_template().format(items_content=items_str)
        
        # Call LLM
        response_text = await llm.generate_text(prompt)
        
        # Parse Batch Response
        results = []
        # Expecting format: ID: <uuid> | SCORE: <0-10> ...
        lines = response_text.strip().split('\n')
        
        # Create a lookup map for items to easily retrieve them by ID if needed (though we parse ID from text)
        item_map = {str(item.id): item for item in items}
        
        for line in lines:
            line = line.strip()
            if not line or "ID:" not in line:
                continue
                
            try:
                # Extract ID first
                parts = [p.strip() for p in line.split('|')]
                id_part = next((p for p in parts if p.startswith("ID:")), None)
                if not id_part:
                    continue
                    
                item_id = id_part.replace("ID:", "").strip()
                original_item = item_map.get(item_id)
                
                if original_item:
                    result = self.parse_line(original_item, line)
                    if result:
                        results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse batch line: {line} - {e}")
                continue
                
        return results

    @abstractmethod
    def get_batch_prompt_template(self) -> str:
        pass
        
    @abstractmethod
    def parse_line(self, item: IngestedItem, line: str) -> EvaluationResult:
        pass

    # Deprecated single evaluate (optional to keep for fallback, but we will use batch main)
    async def evaluate(self, item: IngestedItem) -> EvaluationResult:
        return (await self.evaluate_batch([item]))[0]
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        return "" # Deprecated
    @abstractmethod
    def parse_response(self, item: IngestedItem, text: str) -> EvaluationResult:
        return EvaluationResult(item_id=item.id, persona="UNKNOWN", score=0, decision="DISCARD", reasoning="", details={})


class GenAiEvaluator(BaseEvaluator):
    def get_persona_name(self) -> str:
        return "GENAI_NEWS"

    def get_batch_prompt_template(self) -> str:
        return """
You are an expert AI Editor.
Analyze the following list of content items.

GUIDELINES:
- Select items relevant to a Generative AI Engineer.
- STRICTLY DISCARD generic non-technical news.
- IGNORE duplicates.

INPUT ITEMS:
{items_content}

OUTPUT FORMAT:
For EACH item, output a SINGLE LINE in this exact format:
ID: <UUID> | SCORE: <0-10> | DECISION: <KEEP/DISCARD> | INSIGHT: <One sentence summary of the main technical takeaway or key finding>

Output ONLY these lines.
"""

    def parse_line(self, item: IngestedItem, line: str) -> EvaluationResult:
        data = {}
        parts = [p.strip() for p in line.split('|')]
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                data[k.strip().upper()] = v.strip()
        
        score = float(data.get('SCORE', 0))
        decision = data.get('DECISION', 'DISCARD').upper()
        if 'KEEP' in decision and score >= 5: 
            decision = 'KEEP'
        else: 
            decision = 'DISCARD'
            
        return EvaluationResult(
            item_id=item.id,
            persona=self.get_persona_name(),
            score=score,
            decision=decision,
            reasoning=data.get('INSIGHT', ''),
            details={'raw_line': line}
        )
        
    def get_prompt_template(self) -> str: return ""
    def parse_response(self, item, text): return None

class ProductEvaluator(BaseEvaluator):
    def get_persona_name(self) -> str:
        return "PRODUCT_IDEAS"

    def get_batch_prompt_template(self) -> str:
        return """
You are a Product Scout. Analyze the items.
Look for: Startup ideas, unaddressed problems, or market gaps.

INPUT ITEMS:
{items_content}

OUTPUT FORMAT:
For EACH item, output a SINGLE LINE in this exact format:
ID: <UUID> | SCORE: <0-10> | DECISION: <KEEP/DISCARD> | INSIGHT: <One sentence describing the core problem or opportunity>

Output ONLY these lines.
"""

    def parse_line(self, item: IngestedItem, line: str) -> EvaluationResult:
        data = {}
        parts = [p.strip() for p in line.split('|')]
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                data[k.strip().upper()] = v.strip()
        
        score = float(data.get('SCORE', 0))
        decision = data.get('DECISION', 'DISCARD').upper()
        if 'KEEP' in decision and score >= 5: 
            decision = 'KEEP'
        else: 
            decision = 'DISCARD'

        return EvaluationResult(
            item_id=item.id,
            persona=self.get_persona_name(),
            score=score,
            decision=decision,
            reasoning=data.get('INSIGHT', ''),
            details={'raw_line': line}
        )
    def get_prompt_template(self) -> str: return ""
    def parse_response(self, item, text): return None

class FinanceEvaluator(BaseEvaluator):
    def get_persona_name(self) -> str:
        return "FINANCIAL_ANALYSIS"

    def get_batch_prompt_template(self) -> str:
        return """
You are a Financial Analyst. Analyze the items.
Look for: Revenue, Funding, IPOs, Market Data.

INPUT ITEMS:
{items_content}

OUTPUT FORMAT:
For EACH item, output a SINGLE LINE in this exact format:
ID: <UUID> | SCORE: <0-10> | DECISION: <KEEP/DISCARD> | INSIGHT: <Key financial numbers or status update>

Output ONLY these lines.
"""

    def parse_line(self, item: IngestedItem, line: str) -> EvaluationResult:
        data = {}
        parts = [p.strip() for p in line.split('|')]
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                data[k.strip().upper()] = v.strip()
        
        score = float(data.get('SCORE', 0))
        decision = data.get('DECISION', 'DISCARD').upper()
        if 'KEEP' in decision and score >= 5: 
            decision = 'KEEP'
        else: 
            decision = 'DISCARD'

        return EvaluationResult(
            item_id=item.id,
            persona=self.get_persona_name(),
            score=score,
            decision=decision,
            reasoning=data.get('INSIGHT', ''),
            details={'raw_line': line}
        )
    def get_prompt_template(self) -> str: return ""
    def parse_response(self, item, text): return None


class EvaluatorFactory:
    @staticmethod
    def get_evaluator(persona: str) -> BaseEvaluator:
        if persona == "GENAI_NEWS":
            return GenAiEvaluator()
        elif persona == "PRODUCT_IDEAS":
            return ProductEvaluator()
        elif persona == "FINANCIAL_ANALYSIS":
            return FinanceEvaluator()
        else:
            raise ValueError(f"Unknown persona: {persona}")
