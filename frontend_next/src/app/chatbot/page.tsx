"use client";

import { MedicalChat } from "@/components/medical/medical-chat";
import { RouteGuard } from "@/components/auth/route-guard";
import { useAuth } from "@/contexts/auth-context";

export default function ChatbotPage() {
  const { user } = useAuth();

  return (
    <RouteGuard requireAuth fallbackLabel="Sign in to open your chats.">
      <div className="min-h-[calc(100svh-64px)] bg-[var(--brand-bg)]">
        <MedicalChat userName={user?.name ?? "there"} />
      </div>
    </RouteGuard>
  );
}
