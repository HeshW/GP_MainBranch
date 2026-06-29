"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Formik } from "formik";
import * as Yup from "yup";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SectionHeading } from "@/components/ui/section-heading";
import { useAuth } from "@/contexts/auth-context";
import { RouteGuard } from "@/components/auth/route-guard";

const loginSchema = Yup.object({
  email: Yup.string().email("Enter a valid email").required("Email is required"),
  password: Yup.string().required("Password is required"),
});

export default function LoginPage() {
  const router = useRouter();
  const [nextPath, setNextPath] = useState("/chatbot");
  const { login } = useAuth();

  useEffect(() => {
    const next = new URLSearchParams(window.location.search).get("next");
    if (next?.startsWith("/")) setNextPath(next);
  }, []);

  return (
    <RouteGuard guestOnly fallbackLabel="You're already signed in.">
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Account"
        title="Sign in to Nabda"
        description="Access your saved medical assistant conversations across sessions."
      />

      <Card className="mt-8">
        <Formik
          initialValues={{ email: "", password: "", formError: "" }}
          validationSchema={loginSchema}
          onSubmit={async (values, helpers) => {
            helpers.setStatus(undefined);
            try {
              await login({ email: values.email, password: values.password });
              router.push(nextPath);
            } catch (error) {
              helpers.setStatus(error instanceof Error ? error.message : "Login failed.");
            }
          }}
        >
          {({ values, errors, touched, handleChange, handleBlur, handleSubmit, isSubmitting, status }) => (
            <form onSubmit={handleSubmit} className="grid gap-4">
              <label>
                <span className="mb-2 block text-sm font-medium text-slate-700">Email</span>
                <Input name="email" type="email" value={values.email} onChange={handleChange} onBlur={handleBlur} placeholder="you@example.com" />
                {touched.email && errors.email ? <span className="mt-2 block text-xs font-medium text-rose-600">{errors.email}</span> : null}
              </label>
              <label>
                <span className="mb-2 block text-sm font-medium text-slate-700">Password</span>
                <Input name="password" type="password" value={values.password} onChange={handleChange} onBlur={handleBlur} placeholder="Your password" />
                {touched.password && errors.password ? <span className="mt-2 block text-xs font-medium text-rose-600">{errors.password}</span> : null}
              </label>
              {status ? <p className="rounded-2xl bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">{status}</p> : null}
              <div className="flex justify-end pt-2">
                <Button type="submit" disabled={isSubmitting}>
                  Sign in
                </Button>
              </div>
              <p className="text-sm text-[var(--brand-muted)]">
                New here?{" "}
                <Link className="font-semibold text-[var(--brand-primary)]" href="/register">
                  Create an account
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
