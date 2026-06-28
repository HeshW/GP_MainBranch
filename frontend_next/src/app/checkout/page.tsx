"use client";

import { useRouter } from "next/navigation";
import { Formik } from "formik";
import * as Yup from "yup";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SectionHeading } from "@/components/ui/section-heading";
import { useCart } from "@/contexts/cart-context";
import { useOrders } from "@/contexts/orders-context";
import { RouteGuard } from "@/components/auth/route-guard";

const checkoutSchema = Yup.object({
  name: Yup.string().min(2).required("Name is required"),
  email: Yup.string().email("Enter a valid email").required("Email is required"),
  address: Yup.string().min(8).required("Shipping address is required"),
  city: Yup.string().min(2).required("City is required"),
  zip: Yup.string().min(4).required("ZIP code is required"),
});

export default function CheckoutPage() {
  const router = useRouter();
  const { items, subtotal, clearCart } = useCart();
  const { createOrder } = useOrders();
  const shipping = items.length ? 18 : 0;
  const total = subtotal + shipping;

  if (items.length === 0) {
    return (
      <div className="nabda-soft-section">
      <div className="mx-auto max-w-3xl px-4 py-20 sm:px-6 lg:px-8">
        <Card className="text-center">
          <h1 className="text-2xl font-semibold text-[var(--brand-heading)]">No care request yet</h1>
          <p className="mt-3 text-sm leading-6 text-slate-600">Start with the chatbot or doctor finder before creating a follow-up request.</p>
        </Card>
      </div>
      </div>
    );
  }

  return (
    <RouteGuard requireAuth fallbackLabel="Sign in to complete checkout.">
    <div className="nabda-soft-section">
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Follow-up"
        title="Complete your care request"
        description="Formik handles the form state while Yup keeps validation rules compact and readable."
      />

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_0.8fr]">
        <Card>
          <Formik
            initialValues={{ name: "", email: "", address: "", city: "", zip: "" }}
            validationSchema={checkoutSchema}
            onSubmit={(values, helpers) => {
              const order = createOrder({
                customerName: values.name,
                email: values.email,
                shippingAddress: `${values.address}, ${values.city}, ${values.zip}`,
                items,
                total,
              });

              clearCart();
              helpers.resetForm();
              router.push(`/orders?order=${order.id}`);
            }}
          >
            {({ values, errors, touched, handleChange, handleBlur, handleSubmit, isSubmitting }) => (
              <form onSubmit={handleSubmit} className="grid gap-4 md:grid-cols-2">
                {[
                  ["name", "Full name", "text"],
                  ["email", "Email", "email"],
                  ["address", "Shipping address", "text"],
                  ["city", "City", "text"],
                  ["zip", "ZIP code", "text"],
                ].map(([field, label, type]) => (
                  <label key={field as string} className={field === "address" ? "md:col-span-2" : ""}>
                    <span className="mb-2 block text-sm font-medium text-slate-700">{label}</span>
                    <Input
                      name={field as string}
                      type={type as string}
                      value={values[field as keyof typeof values]}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      placeholder={label}
                    />
                    {touched[field as keyof typeof touched] && errors[field as keyof typeof errors] ? (
                      <span className="mt-2 block text-xs font-medium text-rose-600">
                        {errors[field as keyof typeof errors]}
                      </span>
                    ) : null}
                  </label>
                ))}

                <div className="md:col-span-2 mt-2 flex justify-end">
                  <Button type="submit" disabled={isSubmitting}>
                    Submit request
                  </Button>
                </div>
              </form>
            )}
          </Formik>
        </Card>

        <Card className="h-fit">
          <h2 className="text-xl font-semibold text-[var(--brand-heading)]">Request summary</h2>
          <div className="mt-5 space-y-3 text-sm text-slate-600">
            <div className="flex items-center justify-between"><span>Items</span><span>{items.length}</span></div>
            <div className="flex items-center justify-between"><span>Subtotal</span><span>${subtotal.toFixed(2)}</span></div>
            <div className="flex items-center justify-between"><span>Shipping</span><span>${shipping.toFixed(2)}</span></div>
            <div className="flex items-center justify-between border-t border-[var(--brand-border)] pt-3 text-base font-semibold text-[var(--brand-heading)]"><span>Total</span><span>${total.toFixed(2)}</span></div>
          </div>
        </Card>
      </div>
    </div>
    </div>
    </RouteGuard>
  );
}
