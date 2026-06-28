"use client";

import { MedicalChat } from "@/components/medical/medical-chat";
import { useAuth } from "@/contexts/auth-context";

export default function ChatbotPage() {
  const { user } = useAuth();

  return (
    <div className="min-h-[calc(100svh-64px)] bg-[var(--brand-bg)]">
      <MedicalChat userName={user?.name ?? "there"} />
    </div>
  );
}
