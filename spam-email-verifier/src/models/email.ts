export class Email {
    address: string;
    status: string;

    constructor(address: string, status: string) {
        this.address = address;
        this.status = status;
    }
}