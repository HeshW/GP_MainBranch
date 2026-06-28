import { Suspense } from "react";
import { OrdersClient } from "./orders-client";

export default function OrdersPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">Loading orders...</div>}>
      <OrdersClient />
    </Suspense>
  );
}
