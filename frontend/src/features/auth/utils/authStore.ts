import type { User } from '../types/authTypes'
import { AUTH_KEYS } from './constants'

/**
 * Utility class for local storage & auth session persistence
 */
export class AuthStore {
  static getAccessToken(): string | null {
    return localStorage.getItem(AUTH_KEYS.ACCESS_TOKEN)
  }

  static setAccessToken(token: string): void {
    localStorage.setItem(AUTH_KEYS.ACCESS_TOKEN, token)
  }

  static getRefreshToken(): string | null {
    return localStorage.getItem(AUTH_KEYS.REFRESH_TOKEN)
  }

  static setRefreshToken(token: string): void {
    localStorage.setItem(AUTH_KEYS.REFRESH_TOKEN, token)
  }

  static getUser(): User | null {
    try {
      const data = localStorage.getItem(AUTH_KEYS.USER_DATA)
      return data ? (JSON.parse(data) as User) : null
    } catch {
      return null
    }
  }

  static setUser(user: User): void {
    localStorage.setItem(AUTH_KEYS.USER_DATA, JSON.stringify(user))
  }

  static setSession(accessToken: string, user: User, refreshToken?: string): void {
    this.setAccessToken(accessToken)
    this.setUser(user)
    if (refreshToken) {
      this.setRefreshToken(refreshToken)
    }
  }

  static clearSession(): void {
    localStorage.removeItem(AUTH_KEYS.ACCESS_TOKEN)
    localStorage.removeItem(AUTH_KEYS.REFRESH_TOKEN)
    localStorage.removeItem(AUTH_KEYS.USER_DATA)
  }

  static getRememberedEmail(): string {
    return localStorage.getItem(AUTH_KEYS.REMEMBERED_EMAIL) || ''
  }

  static setRememberedEmail(email: string, remember: boolean): void {
    if (remember) {
      localStorage.setItem(AUTH_KEYS.REMEMBERED_EMAIL, email)
    } else {
      localStorage.removeItem(AUTH_KEYS.REMEMBERED_EMAIL)
    }
  }

  static isAuthenticated(): boolean {
    return Boolean(this.getAccessToken() && this.getUser())
  }
}
