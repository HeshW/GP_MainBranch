"use client";

import { useRouter } from "next/navigation";
import { Formik } from "formik";
import * as Yup from "yup";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SectionHeading } from "@/components/ui/section-heading";
import { useAuth } from "@/contexts/auth-context";
import { RouteGuard } from "@/components/auth/route-guard";

const loginSchema = Yup.object({
  name: Yup.string().min(2).required("Name is required"),
  email: Yup.string().email("Enter a valid email").required("Email is required"),
});

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();

  return (
    <RouteGuard guestOnly fallbackLabel="You're already signed in.">
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6 lg:px-8">
      <SectionHeading
        eyebrow="Account"
        title="Sign in with the demo form"
        description="Formik and Yup keep the form implementation tidy while the auth state persists locally in the browser."
      />

      <Card className="mt-8">
        <Formik
          initialValues={{ name: "", email: "" }}
          validationSchema={loginSchema}
          onSubmit={(values) => {
            login({ name: values.name, email: values.email });
            router.push("/");
          }}
        >
          {({ values, errors, touched, handleChange, handleBlur, handleSubmit, isSubmitting }) => (
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
              <div className="flex justify-end pt-2">
                <Button type="submit" disabled={isSubmitting}>
                  Enter dashboard
                </Button>
              </div>
            </form>
          )}
        </Formik>
      </Card>
    </div>
    </RouteGuard>
  );
}
