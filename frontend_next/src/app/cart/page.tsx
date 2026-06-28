"use client";

import Link from "next/link";
import Image from "next/image";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SectionHeading } from "@/components/ui/section-heading";
import { useCart } from "@/contexts/cart-context";

export default function CartPage() {
  const { items, subtotal, updateQuantity, removeFromCart } = useCart();
  const shipping = items.length ? 18 : 0;
  const total = subtotal + shipping;

  return (
    <div className="nabda-soft-section">
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Cart"
        title="Review your selected items"
        description="This cart persists in localStorage and updates instantly when you change quantities or remove a product."
      />

      {items.length === 0 ? (
        <Card className="mt-8 text-center">
          <h2 className="text-2xl font-semibold text-[var(--brand-heading)]">Your cart is empty</h2>
          <p className="mt-3 text-sm leading-6 text-slate-600">Start with the catalog and add a few products to continue.</p>
          <div className="mt-6">
            <Link href="/products">
              <Button>Browse products</Button>
            </Link>
          </div>
        </Card>
      ) : (
        <div className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-4">
            {items.map(({ product, quantity }) => (
            <Card key={product.id} className="flex gap-4 p-4">
                <Image
                  src={product.image}
                  alt={product.title}
                  width={96}
                  height={96}
                  className="h-24 w-24 rounded-2xl object-cover"
                />
                <div className="flex-1">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <h2 className="text-lg font-semibold text-[var(--brand-heading)]">{product.title}</h2>
                      <p className="mt-1 text-sm text-slate-500">${product.price} each</p>
                    </div>
                    <button className="text-sm font-semibold text-blue-700" onClick={() => removeFromCart(product.id)}>
                      Remove
                    </button>
                  </div>
                  <div className="mt-4 flex items-center gap-3">
                    <button
                      className="h-10 w-10 rounded-full border border-[var(--brand-border)] bg-white text-lg text-[var(--brand-primary)] shadow-sm transition hover:bg-[var(--brand-soft)]"
                      onClick={() => updateQuantity(product.id, Math.max(1, quantity - 1))}
                    >
                      -
                    </button>
                    <span className="min-w-10 text-center text-sm font-semibold">{quantity}</span>
                    <button className="h-10 w-10 rounded-full border border-[var(--brand-border)] bg-white text-lg text-[var(--brand-primary)] shadow-sm transition hover:bg-[var(--brand-soft)]" onClick={() => updateQuantity(product.id, quantity + 1)}>
                      +
                    </button>
                  </div>
                </div>
              </Card>
            ))}
          </div>

            <Card className="h-fit border-[var(--brand-border)] bg-gradient-to-b from-white to-[var(--brand-soft)]">
            <h2 className="text-xl font-semibold text-[var(--brand-heading)]">Order summary</h2>
            <div className="mt-5 space-y-3 text-sm text-slate-600">
              <div className="flex items-center justify-between"><span>Subtotal</span><span>${subtotal.toFixed(2)}</span></div>
              <div className="flex items-center justify-between"><span>Shipping</span><span>${shipping.toFixed(2)}</span></div>
              <div className="flex items-center justify-between border-t border-[var(--brand-border)] pt-3 text-base font-semibold text-[var(--brand-heading)]"><span>Total</span><span>${total.toFixed(2)}</span></div>
            </div>
            <Link href="/checkout" className="mt-6 block">
              <Button className="w-full">Proceed to checkout</Button>
            </Link>
          </Card>
        </div>
      )}
    </div>
    </div>
  );
}
