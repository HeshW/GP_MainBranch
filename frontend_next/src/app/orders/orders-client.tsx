"use client";

import { useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/card";
import { SectionHeading } from "@/components/ui/section-heading";
import { useOrders } from "@/contexts/orders-context";
import { RouteGuard } from "@/components/auth/route-guard";

export function OrdersClient() {
  const searchParams = useSearchParams();
  const highlightedOrderId = searchParams.get("order");
  const { orders } = useOrders();

  return (
    <RouteGuard requireAuth fallbackLabel="Sign in to view your orders.">
    <div className="nabda-soft-section">
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="History"
        title="Track recent care requests"
        description="Requests are stored locally so you can demo the follow-up flow end to end without a backend."
      />

      <div className="mt-8 grid gap-4">
        {orders.length === 0 ? (
          <Card>
            <p className="text-sm text-slate-600">No care requests yet. Complete a follow-up form to populate this page.</p>
          </Card>
        ) : (
          orders.map((order) => (
            <Card
              key={order.id}
                      className={highlightedOrderId === order.id ? "border-[var(--brand-border-strong)] ring-4 ring-blue-500/10" : ""}
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                          <p className="text-sm font-semibold text-[var(--brand-primary)]">{order.status}</p>
                          <h2 className="mt-2 text-xl font-semibold text-[var(--brand-heading)]">{order.id}</h2>
                  <p className="mt-1 text-sm text-slate-500">{new Date(order.createdAt).toLocaleString()}</p>
                </div>
                <div className="text-right text-sm text-slate-500">
                  <p>{order.customerName}</p>
                  <p>{order.email}</p>
                </div>
              </div>

              <div className="mt-5 grid gap-3 text-sm text-slate-600 md:grid-cols-[1fr_auto] md:items-center">
                <p>{order.shippingAddress}</p>
                <p className="font-semibold text-[var(--brand-heading)]">${order.total.toFixed(2)}</p>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {order.items.map((item) => (
                  <span key={item.product.id} className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                    {item.product.title} × {item.quantity}
                  </span>
                ))}
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
    </div>
    </RouteGuard>
  );
}
