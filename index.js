function calculateTotal(price, quantity, hasDiscount) {
  const subtotal = price * quantity;
  const discount = hasDiscount ? subtotal * 0.1 : 0;
  const discountedSubtotal = subtotal - discount;
  const tax = discountedSubtotal * 0.05;

  return discountedSubtotal + tax;
}

function calculateSimpleInterest(principal, rate) {
  const timeInYears = 1;
  return (principal * rate * timeInYears) / 100;
}
