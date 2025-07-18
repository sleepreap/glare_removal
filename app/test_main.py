import subprocess
import aiohttp
import asyncio

DOCKER_IMAGE = "htx-app"
DOCKER_CONTAINER_NAME = "htx-test-container"
DOCKER_PORT = "4000"
API_URL = f"http://localhost:{DOCKER_PORT}"
TEST_IMAGE = "test1.png"

async def start_docker():
    subprocess.run(f"docker rm -f {DOCKER_CONTAINER_NAME}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Starting Docker container...")

    subprocess.run(
        f"docker run -d -p {DOCKER_PORT}:{DOCKER_PORT} --name {DOCKER_CONTAINER_NAME} {DOCKER_IMAGE}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Poll until /ping works or timeout after 30s
    async with aiohttp.ClientSession() as session:
        for i in range(30):
            try:
                async with session.get(f"{API_URL}/ping") as resp:
                    if resp.status == 200:
                        print("Server is up!")
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(1)
        raise TimeoutError("Server did not become ready in time.")

def stop_docker():
    print("Stopping Docker container...")
    subprocess.run(f"docker stop {DOCKER_CONTAINER_NAME}", shell=True)
    subprocess.run(f"docker rm {DOCKER_CONTAINER_NAME}", shell=True)
    print("Docker container stopped.")

async def test_ping():
    print("Testing /ping...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/ping") as resp:
            assert resp.status == 200
            json_data = await resp.json()
            assert json_data == {"message": "pong"}
    print("/ping passed")

async def test_infer():
    print("Testing /infer...")
    async with aiohttp.ClientSession() as session:
        with open(TEST_IMAGE, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("image", f, filename=TEST_IMAGE, content_type="image/png")

            async with session.post(f"{API_URL}/infer", data=data) as resp:
                assert resp.status == 200
                json_data = await resp.json()
                assert "image" in json_data
    print("/infer passed")

async def main():
    try:
        await start_docker()
        await test_ping()
        await test_infer()
    finally:
        stop_docker()

if __name__ == "__main__":
    asyncio.run(main())
