"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Formik } from "formik";
import * as Yup from "yup";
import { RouteGuard } from "@/components/auth/route-guard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SectionHeading } from "@/components/ui/section-heading";
import { useAuth } from "@/contexts/auth-context";

const registerSchema = Yup.object({
  name: Yup.string().min(2, "Name is too short").required("Name is required"),
  email: Yup.string().email("Enter a valid email").required("Email is required"),
  password: Yup.string().min(8, "Use at least 8 characters").required("Password is required"),
});

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  return (
    <RouteGuard guestOnly fallbackLabel="You're already signed in.">
      <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Account"
          title="Create your Nabda account"
          description="Your chats are stored locally on this backend and only shown to your account."
        />

        <Card className="mt-8">
          <Formik
            initialValues={{ name: "", email: "", password: "" }}
            validationSchema={registerSchema}
            onSubmit={async (values, helpers) => {
              helpers.setStatus(undefined);
              try {
                await register(values);
                router.push("/chatbot");
              } catch (error) {
                helpers.setStatus(error instanceof Error ? error.message : "Registration failed.");
              }
            }}
          >
            {({ values, errors, touched, handleChange, handleBlur, handleSubmit, isSubmitting, status }) => (
              <form onSubmit={handleSubmit} className="grid gap-4">
                <label>
                  <span className="mb-2 block text-sm font-medium text-slate-700">Name</span>
                  <Input name="name" value={values.name} onChange={handleChange} onBlur={handleBlur} placeholder="Your name" />
                  {touched.name && errors.name ? <span className="mt-2 block text-xs font-medium text-rose-600">{errors.name}</span> : null}
                </label>
                <label>
                  <span className="mb-2 block text-sm font-medium text-slate-700">Email</span>
                  <Input name="email" type="email" value={values.email} onChange={handleChange} onBlur={handleBlur} placeholder="you@example.com" />
                  {touched.email && errors.email ? <span className="mt-2 block text-xs font-medium text-rose-600">{errors.email}</span> : null}
                </label>
                <label>
                  <span className="mb-2 block text-sm font-medium text-slate-700">Password</span>
                  <Input name="password" type="password" value={values.password} onChange={handleChange} onBlur={handleBlur} placeholder="At least 8 characters" />
                  {touched.password && errors.password ? <span className="mt-2 block text-xs font-medium text-rose-600">{errors.password}</span> : null}
                </label>
                {status ? <p className="rounded-2xl bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">{status}</p> : null}
                <div className="flex justify-end pt-2">
                  <Button type="submit" disabled={isSubmitting}>
                    Create account
                  </Button>
                </div>
                <p className="text-sm text-[var(--brand-muted)]">
                  Already registered?{" "}
                  <Link className="font-semibold text-[var(--brand-primary)]" href="/login">
                    Sign in
                  </Link>
                </p>
              </form>
            )}
          </Formik>
        </Card>
      </div>
    </RouteGuard>
  );
}
