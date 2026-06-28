"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { readStorage, writeStorage } from "@/lib/storage";
import type { Product } from "@/lib/catalog";

export type OrderItem = {
  product: Product;
  quantity: number;
};

export type Order = {
  id: string;
  createdAt: string;
  customerName: string;
  email: string;
  shippingAddress: string;
  items: OrderItem[];
  total: number;
  status: "Processing" | "Paid" | "Shipped";
};

type OrdersContextValue = {
  orders: Order[];
  createOrder: (order: Omit<Order, "id" | "createdAt" | "status">) => Order;
};

const OrdersContext = createContext<OrdersContextValue | undefined>(undefined);

export function OrdersProvider({ children }: { children: React.ReactNode }) {
  const [orders, setOrders] = useState<Order[]>(() =>
    readStorage<Order[]>("next-ecomm-orders", []),
  );

  useEffect(() => {
    writeStorage("next-ecomm-orders", orders);
  }, [orders]);

  function createOrder(order: Omit<Order, "id" | "createdAt" | "status">) {
    const nextOrder: Order = {
      ...order,
      id: `ORD-${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
      createdAt: new Date().toISOString(),
      status: "Processing",
    };

    setOrders((current) => [nextOrder, ...current]);
    return nextOrder;
  }

  const value = useMemo(() => ({ orders, createOrder }), [orders]);

  return <OrdersContext.Provider value={value}>{children}</OrdersContext.Provider>;
}

export function useOrders() {
  const context = useContext(OrdersContext);
  if (!context) {
    throw new Error("useOrders must be used within OrdersProvider");
  }
  return context;
}
