import json
from app.schemas.ai import AITherapyPlanResponse

class TherapyEngine:
    """Uses Gemini API to generate a professional therapy/treatment plan from a diagnosis."""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key_valid = bool(gemini_api_key and "AIza" in gemini_api_key)
        self._provider: Optional[GeminiProvider] = None
        
        if self.api_key_valid:
            self._provider = GeminiProvider(api_key=gemini_api_key, model_name=model_name)
        else:
            logger.warning("Invalid or missing GEMINI_API_KEY. TherapyEngine will operate in fallback mode.")

    async def generate_therapy(self, diagnosis: Dict[str, Any], patient_info: str = "") -> Dict[str, Any]:
        """Asynchronously generate professional therapy plan recommendations using structured output."""
        findings = diagnosis.get("findings", [])
        if not findings:
            return {
                "therapy_plan": "⚠️ **ملاحظة:** لا توجد نتائج تشخيص غير طبيعية تتطلب خطة علاج مستعجلة، يُنصح بالمتابعة الطبية الدورية."
            }
            
        findings_text = ""
        for f in findings:
            findings_text += f"- **{f.get('condition')}** (الخطورة: {f.get('severity')})\n"
            findings_text += f"  الدليل: {f.get('evidence')}\n"

        if not self.api_key_valid or not self._provider:
            return {
                "therapy_plan": (
                    "⚠️ **تنبيه:** وضع المعاينة فقط (Fallback Mode).\n\n"
                    f"تم اكتشاف {len(findings)} ملاحظات طبية. "
                    "يرجى مراجعة طبيب مختص فوراً لتقييم هذه النتائج وتحديد خطة العلاج المناسبة."
                )
            }

        prompt = f"""
        أنت استشاري طبي خبير. بناءً على التشخيص المبدئي الموضح أدناه، قم بإعداد خطة توصيات طبية احترافية وشاملة باللغة العربية.

        [نتائج التشخيص الطبي]
        {findings_text}
        
        [معلومات إضافية عن المريض]
        {patient_info if patient_info else "غير متوفرة"}
        """
        
        system_instruction = (
            "You are a helpful and professional Medical Consultant AI. "
            "You must provide medical recommendations in Arabic. "
            "Your response must be a valid JSON object matching the requested schema. "
            "Ensure the output reflects clinical excellence and professional tone."
        )

        try:
            response_json = await self._provider.generate_content(
                prompt,
                system_instruction=system_instruction,
                response_model=AITherapyPlanResponse
            )
            
            structured_data = json.loads(response_json)
            
            # Reconstruct therapy_plan as markdown for backward compatibility or display
            recommendations = "\n".join([f"- **{r['category']}**: {r['description']} ({r['urgency']})" for r in structured_data['recommendations']])
            lifestyle = structured_data['lifestyle_advice']
            emergency = "\n".join([f"- {s}" for s in structured_data['emergency_signs']])
            
            therapy_markdown = (
                f"{structured_data['disclaimer']}\n\n"
                f"### تحليل الحالة\n{structured_data['clinical_analysis']}\n\n"
                f"### التوصيات العلاجية\n{recommendations}\n\n"
                f"### نمط الحياة والتغذية\n{lifestyle}\n\n"
                f"### علامات تستوجب الطوارئ\n{emergency}"
            )

            return {
                "therapy_plan": therapy_markdown,
                "structured_therapy": structured_data
            }
        except Exception as e:
            logger.error(f"Therapy generation failed: {e}")
            return {"therapy_plan": "❌ تعذر توليد خطة العلاج حالياً بشكل منظم. يرجى مراجعة سجلات النظام."}
