"use client";

import { createContext, useContext, useMemo, useState } from "react";
import { readStorage, removeStorage, writeStorage } from "@/lib/storage";

export type User = {
  name: string;
  email: string;
};

type AuthContextValue = {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isReady: boolean;
  login: (user: User) => void;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() =>
    readStorage<User | null>("next-ecomm-user", null),
  );
  const [token, setToken] = useState<string | null>(() =>
    readStorage<string | null>("next-ecomm-token", null),
  );
  const [isReady] = useState(true);

  function login(nextUser: User) {
    const nextToken = `demo-${Date.now()}`;
    setUser(nextUser);
    setToken(nextToken);
    writeStorage("next-ecomm-user", nextUser);
    writeStorage("next-ecomm-token", nextToken);
  }

  function logout() {
    setUser(null);
    setToken(null);
    removeStorage("next-ecomm-user");
    removeStorage("next-ecomm-token");
  }

  const value = useMemo(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(token),
      isReady,
      login,
      logout,
    }),
    [isReady, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
