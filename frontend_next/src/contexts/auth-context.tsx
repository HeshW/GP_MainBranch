"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { fetchCurrentUser, loginUser, registerUser, type AuthUser } from "@/lib/api";
import { readStorage, removeStorage, writeStorage } from "@/lib/storage";

export type User = AuthUser;

type AuthContextValue = {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isReady: boolean;
  login: (credentials: { email: string; password: string }) => Promise<void>;
  register: (credentials: { name: string; email: string; password: string }) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    removeStorage("next-ecomm-user");
    removeStorage("next-ecomm-token");
  }, []);

  useEffect(() => {
    let active = true;

    queueMicrotask(() => {
      if (!active) {
        return;
      }

      const storedUser = readStorage<User | null>("next-ecomm-user", null);
      const storedToken = readStorage<string | null>("next-ecomm-token", null);

      if (!storedToken) {
        setIsReady(true);
        return;
      }

      setUser(storedUser);
      setToken(storedToken);

      fetchCurrentUser(storedToken)
        .then((nextUser) => {
          if (!active) return;
          setUser(nextUser);
          writeStorage("next-ecomm-user", nextUser);
        })
        .catch(() => {
          if (!active) return;
          logout();
        })
        .finally(() => {
          if (active) setIsReady(true);
        });
    });

    return () => {
      active = false;
    };
  }, [logout]);

  const storeSession = useCallback((nextUser: User, nextToken: string) => {
    setUser(nextUser);
    setToken(nextToken);
    writeStorage("next-ecomm-user", nextUser);
    writeStorage("next-ecomm-token", nextToken);
  }, []);

  const login = useCallback(async (credentials: { email: string; password: string }) => {
    const response = await loginUser(credentials);
    storeSession(response.user, response.access_token);
  }, [storeSession]);

  const register = useCallback(async (credentials: { name: string; email: string; password: string }) => {
    const response = await registerUser(credentials);
    storeSession(response.user, response.access_token);
  }, [storeSession]);

  const value = useMemo(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(token),
      isReady,
      login,
      register,
      logout,
    }),
    [isReady, login, register, logout, token, user],
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
