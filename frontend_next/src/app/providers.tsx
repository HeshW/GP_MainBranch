"use client";

import type { ReactNode } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/query-client";
import { AuthProvider } from "@/contexts/auth-context";
import { CartProvider } from "@/contexts/cart-context";
import { OrdersProvider } from "@/contexts/orders-context";
import { PreferencesProvider } from "@/contexts/preferences-context";

export default function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <PreferencesProvider>
        <AuthProvider>
          <CartProvider>
            <OrdersProvider>{children}</OrdersProvider>
          </CartProvider>
        </AuthProvider>
      </PreferencesProvider>
    </QueryClientProvider>
  );
}
