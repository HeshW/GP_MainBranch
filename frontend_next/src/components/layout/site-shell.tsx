import type { ReactNode } from "react";
import { FloatingMedicalChat } from "@/components/medical/floating-medical-chat";
import { Navbar } from "./navbar";
import { Footer } from "./footer";

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-[var(--brand-bg)]">
      <Navbar />
      <main className="flex-1">{children}</main>
      <FloatingMedicalChat />
      <Footer />
    </div>
  );
}
