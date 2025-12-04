import { EmailController } from '../src/controllers/emailController';
import { Verifier } from '../src/services/verifier';
import { validateEmail } from '../src/validators/emailValidator';

describe('Email Verification Tests', () => {
    let emailController: EmailController;
    let verifier: Verifier;

    beforeEach(() => {
        verifier = new Verifier();
        emailController = new EmailController(verifier);
    });

    test('should validate a correct email format', () => {
        const email = 'test@example.com';
        const result = validateEmail(email);
        expect(result.isValid).toBe(true);
    });

    test('should invalidate an incorrect email format', () => {
        const email = 'invalid-email';
        const result = validateEmail(email);
        expect(result.isValid).toBe(false);
    });

    test('should check if an email is spam', async () => {
        const email = 'spam@example.com';
        const isSpam = await verifier.checkSpam(email);
        expect(isSpam).toBe(true);
    });

    test('should verify email and return status', async () => {
        const email = 'test@example.com';
        const response = await emailController.verifyEmail(email);
        expect(response.status).toBe('verified');
    });

    test('should return verification status', async () => {
        const email = 'test@example.com';
        await emailController.verifyEmail(email);
        const status = await emailController.getVerificationStatus(email);
        expect(status).toBe('verified');
    });
});