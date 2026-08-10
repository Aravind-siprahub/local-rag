import { z } from 'zod'

/**
 * Password validation regex patterns
 */
const uppercaseRegex = /[A-Z]/
const lowercaseRegex = /[a-z]/
const numberRegex = /[0-9]/
const specialCharRegex = /[^A-Za-z0-9]/

/**
 * Reusable strong password schema
 */
export const passwordSchema = z
  .string()
  .min(8, 'Password must be at least 8 characters')
  .refine((val) => uppercaseRegex.test(val), {
    message: 'Password must contain at least one uppercase letter',
  })
  .refine((val) => lowercaseRegex.test(val), {
    message: 'Password must contain at least one lowercase letter',
  })
  .refine((val) => numberRegex.test(val), {
    message: 'Password must contain at least one number',
  })
  .refine((val) => specialCharRegex.test(val), {
    message: 'Password must contain at least one special character (@$!%*?&)',
  })

/**
 * Login Form Validation Schema
 */
export const loginSchema = z.object({
  email: z
    .string()
    .min(1, 'Email address is required')
    .email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
  rememberMe: z.boolean().optional().default(false),
})

export type LoginSchemaType = z.infer<typeof loginSchema>

/**
 * Sign Up Form Validation Schema
 */
export const signupSchema = z
  .object({
    fullName: z
      .string()
      .min(1, 'Full name is required')
      .min(2, 'Full name must be at least 2 characters')
      .max(60, 'Full name must not exceed 60 characters'),
    email: z
      .string()
      .min(1, 'Email address is required')
      .email('Please enter a valid email address'),
    password: passwordSchema,
    confirmPassword: z.string().min(1, 'Please confirm your password'),
    acceptTerms: z.boolean().refine((val) => val === true, {
      message: 'You must accept the Terms of Service & Privacy Policy',
    }),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  })

export type SignupSchemaType = z.infer<typeof signupSchema>

/**
 * Forgot Password Schema
 */
export const forgotPasswordSchema = z.object({
  email: z
    .string()
    .min(1, 'Email address is required')
    .email('Please enter a valid email address'),
})

export type ForgotPasswordSchemaType = z.infer<typeof forgotPasswordSchema>
