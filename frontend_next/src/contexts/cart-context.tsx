"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { featuredProducts, type Product } from "@/lib/catalog";
import { readStorage, writeStorage } from "@/lib/storage";

type CartItem = {
  product: Product;
  quantity: number;
};

type CartContextValue = {
  items: CartItem[];
  itemCount: number;
  subtotal: number;
  addToCart: (product: Product, quantity?: number) => void;
  updateQuantity: (productId: number, quantity: number) => void;
  removeFromCart: (productId: number) => void;
  clearCart: () => void;
};

const CartContext = createContext<CartContextValue | undefined>(undefined);

function normalizeCart(items: CartItem[]) {
  return items.filter((item) => item.quantity > 0);
}

export function CartProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CartItem[]>(() =>
    readStorage<CartItem[]>("next-ecomm-cart", []),
  );

  useEffect(() => {
    writeStorage("next-ecomm-cart", items);
  }, [items]);

  function addToCart(product: Product, quantity = 1) {
    setItems((current) => {
      const existing = current.find((entry) => entry.product.id === product.id);
      if (existing) {
        return current.map((entry) =>
          entry.product.id === product.id
            ? { ...entry, quantity: entry.quantity + quantity }
            : entry,
        );
      }

      return [...current, { product, quantity }];
    });
  }

  function updateQuantity(productId: number, quantity: number) {
    setItems((current) =>
      normalizeCart(
        current.map((entry) =>
          entry.product.id === productId ? { ...entry, quantity } : entry,
        ),
      ),
    );
  }

  function removeFromCart(productId: number) {
    setItems((current) => current.filter((entry) => entry.product.id !== productId));
  }

  function clearCart() {
    setItems([]);
  }

  const subtotal = useMemo(
    () => items.reduce((sum, item) => sum + item.product.price * item.quantity, 0),
    [items],
  );

  const itemCount = useMemo(
    () => items.reduce((sum, item) => sum + item.quantity, 0),
    [items],
  );

  const value = useMemo(
    () => ({
      items,
      itemCount,
      subtotal,
      addToCart,
      updateQuantity,
      removeFromCart,
      clearCart,
    }),
    [itemCount, items, subtotal],
  );

  return <CartContext.Provider value={value}>{children}</CartContext.Provider>;
}

export function useCart() {
  const context = useContext(CartContext);
  if (!context) {
    throw new Error("useCart must be used within CartProvider");
  }
  return context;
}

export function getSeedCartItems() {
  return featuredProducts.slice(0, 2).map((product, index) => ({
    product,
    quantity: index + 1,
  }));
}
