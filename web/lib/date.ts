/**
 * Date utility functions for timezone conversion
 */

/**
 * Convert UTC date to KST (Korea Standard Time, UTC+9)
 */
export function toKST(date: Date | string): Date {
  const d = typeof date === 'string' ? new Date(date) : date;
  // Add 9 hours for KST
  return new Date(d.getTime() + 9 * 60 * 60 * 1000);
}

/**
 * Format date to KST string
 */
export function formatKST(date: Date | string, formatStr: string = 'yyyy-MM-dd HH:mm', showSuffix: boolean = true): string {
  const { format } = require('date-fns');
  const kstDate = toKST(date);
  return format(kstDate, formatStr) + (showSuffix ? ' KST' : '');
}

/**
 * Format date only (no time) - for prediction dates which are already in market time
 */
export function formatDateOnly(date: Date | string): string {
  const { format } = require('date-fns');
  const d = typeof date === 'string' ? new Date(date) : date;
  return format(d, 'yyyy-MM-dd');
}
